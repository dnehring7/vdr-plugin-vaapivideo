// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Dirk Nehring <dnehring@gmx.net>
/**
 * @file stream.cpp
 * @brief Codec detection + minimal SPS peek + backend-table lookup helpers.
 */

#include "stream.h"

#include "caps.h"

// C++ standard library
#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

// FFmpeg
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wsign-conversion"
extern "C" {
#include <libavcodec/codec_id.h>
#include <libavcodec/defs.h>
#include <libavutil/intreadwrite.h>
}
#pragma GCC diagnostic pop

// VDR
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wvariadic-macros"
#include <vdr/tools.h>
#pragma GCC diagnostic pop

// ============================================================================
// === Backend table lookup ===
// ============================================================================

auto SelectVideoBackendCap(const VideoStreamInfo &info) noexcept -> bool GpuCaps::* {
    // Linear scan: first row whose codec matches AND bitDepth matches AND profile
    // matches (or is FF_PROFILE_UNKNOWN wildcard) wins. Order rows most-specific-first.
    for (const auto &row : kVideoBackendTable) {
        if (row.codecId != info.codecId) {
            continue;
        }
        if (row.bitDepth != info.bitDepth) {
            continue;
        }
        if (row.profile != AV_PROFILE_UNKNOWN && row.profile != info.profile) {
            continue;
        }
        return row.flag;
    }
    return nullptr;
}

auto IsPassthroughCapable(AVCodecID codec) noexcept -> bool {
    return std::ranges::find(kAudioPassthroughTable, codec) != kAudioPassthroughTable.end();
}

// ============================================================================
// === CODEC DETECTION ===
// ============================================================================

namespace {

/// Accumulated NAL evidence for one codec during a DetectVideoCodec() scan.
/// seenMask bits represent distinct strong-NAL types; lastPos is the buffer offset
/// of the most recent hit, used to tie-break equal hit-counts by preferring
/// evidence later in the buffer (more likely to reflect the current stream than
/// stale bytes from a previous channel in the receiver ring).
struct CodecEvidence {
    uint8_t seenMask{}; ///< Distinct strong-NAL type bits seen so far
    size_t lastPos{};   ///< Buffer offset of the most recent matching NAL (tie-break)

    auto Record(uint8_t bit, size_t pos) noexcept -> void {
        seenMask |= bit;
        lastPos = pos;
    }
};

/// Returns 3 for `00 00 01`, 4 for `00 00 00 01`, 0 otherwise.
/// Shared by DetectVideoCodec() and FindNal() to walk Annex-B NAL boundaries.
[[nodiscard]] constexpr auto AnnexBStartCodeLength(const uint8_t *data, size_t size, size_t offset) noexcept -> size_t {
    if (offset + 3 > size) {
        return 0;
    }
    // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic) -- guarded by offset+3 <= size
    if (data[offset] != 0x00 || data[offset + 1] != 0x00) {
        return 0;
    }
    if (data[offset + 2] == 0x01) {
        return 3;
    }
    if (offset + 4 <= size && data[offset + 2] == 0x00 && data[offset + 3] == 0x01) {
        return 4;
    }
    // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    return 0;
}

} // namespace

auto DetectAudioCodec(std::span<const uint8_t> data) noexcept -> AVCodecID {
    // Linear sync-word scan; first decisive match wins. Test order is intentional --
    // see the AAC-vs-MP2 and AAC-LATM disambiguation comments below.
    if (data.size() < 4) [[unlikely]] {
        return AV_CODEC_ID_NONE;
    }

    const size_t size = data.size();
    const uint8_t *p = data.data();

    for (size_t i = 0; i + 4 <= size; ++i) {
        const uint16_t sync = AV_RB16(p + i);

        // AAC ADTS sync: layer field == 00 (masked 0xFFF6 == 0xFFF0). MP2 uses layer 01/10/11
        // with the looser 0xFFE0 mask, which would also hit every ADTS frame -- test ADTS first.
        if ((sync & 0xFF00) == 0xFF00) [[unlikely]] {
            if ((sync & 0xFFF6) == 0xFFF0) [[unlikely]] {
                return AV_CODEC_ID_AAC;
            }
            // (sync & 0x06) != 0: layer field != 00 (reserved), rejects random 0xFFE? bytes.
            if ((sync & 0xFFE0) == 0xFFE0 && (sync & 0x06) != 0x00) [[likely]] {
                return AV_CODEC_ID_MP2;
            }
        }

        // AAC-LATM/LOAS (ISO 14496-3 sec.1.7.3): 11-bit syncword is too short for single-frame
        // confidence. Confirm with a second sync at pos + 3-byte header + audioMuxLengthBytes
        // (lower 13 bits of the header word).
        if ((sync & 0xFFE0) == 0x56E0) [[unlikely]] {
            const auto frameLen = static_cast<uint16_t>(((sync & 0x1FU) << 8) | AV_RB8(p + i + 2));
            if (frameLen >= 2) {
                const size_t next = i + 3 + frameLen;
                if (next + 2 <= size && (AV_RB16(p + next) & 0xFFE0) == 0x56E0) {
                    return AV_CODEC_ID_AAC_LATM;
                }
            }
        }

        // AC-3 / E-AC-3 share sync 0x0B77 (ATSC A/52). Disambiguated by bsid (5 bits at
        // byte+5 bits 7..3, A/52 sec.A.4.3): bsid <= 10 -> AC-3, 11-15 -> reserved (classified
        // as E-AC-3 to match FFmpeg, but no real stream uses these), 16 -> E-AC-3.
        if (sync == 0x0B77) [[unlikely]] {
            if (i + 5 < size && ((AV_RB8(p + i + 5) >> 3) & 0x1F) > 10) [[unlikely]] {
                return AV_CODEC_ID_EAC3;
            }
            return AV_CODEC_ID_AC3;
        }

        // DTS Coherent Acoustics core (ETSI TS 102 114). 32-bit sync, no ambiguity.
        if (AV_RB32(p + i) == 0x7FFE8001) [[unlikely]] {
            return AV_CODEC_ID_DTS;
        }

        // TrueHD major sync (0xF8726FBA, Dolby TrueHD sec.5.3). Mandatory in the first frame
        // and recurs periodically, so probing for it on early input is reliable.
        if (AV_RB32(p + i) == 0xF8726FBA) [[unlikely]] {
            return AV_CODEC_ID_TRUEHD;
        }
    }

    return AV_CODEC_ID_NONE;
}

auto DetectVideoCodec(std::span<const uint8_t> data) noexcept -> AVCodecID {
    // Single Annex-B pass accumulating per-codec "strong NAL" evidence. Thresholds:
    //   HEVC   -- popcount(seenMask) >= 2 AND (VPS bit set OR popcount(params) >= 2)
    //   H.264  -- popcount(seenMask) >= 2 of {SPS, PPS, IDR}
    //   MPEG-2 -- sequence_header(0xB3) AND (extension(0xB5) OR GOP(0xB8))
    //
    // Cross-codec hazard: MPEG-2 start codes 0xB3/0xB5/0xB8 are legal byte patterns when
    // reinterpreted as H.264/HEVC NAL headers and inflate avc/hevc evidence with phantom
    // hits. Confirmed MPEG-2 invalidates avc/hevc via the !mpegOk gates below.
    const size_t size = data.size();
    if (size < 6) [[unlikely]] {
        return AV_CODEC_ID_NONE;
    }

    const uint8_t *p = data.data();

    CodecEvidence hevc{};
    CodecEvidence avc{};
    CodecEvidence mpeg2{};

    // HEVC seenMask bit layout: 0x01=VPS(type 32), 0x02=SPS(33), 0x04=PPS(34), 0x08=IDR/CRA.
    // Param-set mask = 0x07 covers VPS|SPS|PPS but not the IDR bit.
    constexpr uint8_t kHevcParamMask = 0x07;

    for (size_t i = 0; i + 4 <= size;) {
        // memchr skips the long zero-free runs that dominate compressed video; SIMD on x86_64.
        const auto *found = static_cast<const uint8_t *>(std::memchr(p + i, 0x00, size - i));
        if (!found) [[unlikely]] {
            break;
        }
        i = static_cast<size_t>(found - p);

        if (i + 4 > size) [[unlikely]] {
            break;
        }

        const size_t startCodeLen = AnnexBStartCodeLength(p, size, i);
        if (startCodeLen == 0) {
            ++i;
            continue;
        }

        const size_t nalPos = i + startCodeLen;
        if (nalPos >= size) [[unlikely]] {
            break;
        }

        const uint8_t nalHeader0 = p[nalPos];

        // 0xB3/0xB5/0xB8 all have high bit set -> forbidden_zero_bit=1 in H.264/HEVC, spec-illegal
        // for those codecs, so these are unambiguous MPEG-2 start codes.
        // mpeg2 seenMask: sequence_header(0xB3)=0x01, extension(0xB5)=0x02, GOP(0xB8)=0x04.
        if (nalHeader0 == 0xB3 || nalHeader0 == 0xB5 || nalHeader0 == 0xB8) {
            uint8_t bit = 0x04;
            if (nalHeader0 == 0xB3) {
                bit = 0x01;
            } else if (nalHeader0 == 0xB5) {
                bit = 0x02;
            }
            mpeg2.Record(bit, i);
            // Early exit once the same threshold used at the bottom is satisfied.
            if ((mpeg2.seenMask & 0x01) != 0 && (mpeg2.seenMask & 0x06) != 0) [[unlikely]] {
                break;
            }
        }

        // HEVC NAL header (2 bytes, H.265 sec.7.3.1.2):
        //   byte0: forbidden_zero_bit(1) | nal_unit_type(6) | nuh_layer_id_high(1)
        //   byte1: nuh_layer_id_low(5)   | nuh_temporal_id_plus1(3)
        // Require nuh_layer_id==0 for VPS/SPS/PPS: H.264 byte 0x41 (nal_ref_idc=2, type=1)
        // aliases HEVC type 32 (VPS), but its byte-1 layer bits are random and rarely zero.
        // nuh_temporal_id_plus1==0 is spec-reserved (encodes temporal_id+1), so that's a
        // cheap H.264 false-positive reject too.
        if (nalPos + 1 < size && (nalHeader0 & 0x80) == 0) {
            const uint8_t nalHeader1 = p[nalPos + 1];
            const uint8_t temporalIdPlus1 = nalHeader1 & 0x07;

            if (temporalIdPlus1 != 0) {
                const uint8_t hevcType = (nalHeader0 >> 1) & 0x3F;
                const bool layerIdZero = ((nalHeader0 & 0x01) == 0) && (((nalHeader1 >> 3) & 0x1F) == 0);
                uint8_t bit = 0;

                if (hevcType >= 32 && hevcType <= 34 && layerIdZero) {
                    bit = static_cast<uint8_t>(1U << (hevcType - 32)); // VPS=0x01, SPS=0x02, PPS=0x04
                } else if (hevcType >= 19 && hevcType <= 21 && layerIdZero) {
                    // IDR_W_RADL(19)/IDR_N_LP(20)/CRA(21): gate on a prior param set because
                    // H.264 bytes 0x27/0x28 alias HEVC types 19/20; IDR-only is not enough.
                    if ((hevc.seenMask & kHevcParamMask) != 0) {
                        bit = 0x08;
                    }
                }

                if (bit != 0) {
                    hevc.Record(bit, i);
                }
            }
        }

        // H.264 NAL header (1 byte, sec.7.3.1): forbidden_zero_bit(1)|nal_ref_idc(2)|type(5).
        // Spec requires nal_ref_idc != 0 for SPS(7)/PPS(8)/IDR(5) (sec.7.4.1). This also
        // rejects MPEG-2 start codes 0x01..0x1F whose reinterpreted nal_ref_idc == 0.
        if ((nalHeader0 & 0x80) == 0 && (nalHeader0 & 0x60) != 0) {
            const uint8_t avcType = nalHeader0 & 0x1F;
            uint8_t bit = 0;

            if (avcType == 7) {
                bit = 0x01; // SPS
            } else if (avcType == 8) {
                bit = 0x02; // PPS
            } else if (avcType == 5) {
                bit = 0x04; // IDR
            }

            if (bit != 0) {
                avc.Record(bit, i);
            }
        }

        i = nalPos;
    }

    // HEVC threshold: 2+ distinct NALs total AND (VPS seen OR 2+ distinct param sets).
    // VPS is essentially HEVC-unique (H.264 has no equivalent), so VPS+IDR is sufficient.
    // The disjunction requires richer evidence when VPS is absent.
    const int hevcHits = std::popcount(hevc.seenMask);
    const int hevcParams = std::popcount(static_cast<uint8_t>(hevc.seenMask & kHevcParamMask));
    const bool hevcOk = (hevcHits >= 2) && ((hevc.seenMask & 0x01) != 0 || hevcParams >= 2);

    const bool avcOk = std::popcount(avc.seenMask) >= 2;
    const bool mpegOk = ((mpeg2.seenMask & 0x01) != 0) && ((mpeg2.seenMask & 0x06) != 0);

    // Confirmed MPEG-2 voids avc/hevc (phantom hits from MPEG-2 start codes as NAL headers).
    const bool hevcFinal = hevcOk && !mpegOk;
    const bool avcFinal = avcOk && !mpegOk;

    // Tie-break: more distinct NAL types wins; equal hits -> later buffer position wins.
    AVCodecID best = AV_CODEC_ID_NONE;
    size_t bestPos = 0;
    int bestHits = 0;

    const auto consider = [&](AVCodecID id, bool ok, const CodecEvidence &ev) noexcept -> void {
        if (!ok) {
            return;
        }
        const int h = std::popcount(ev.seenMask);
        if (best == AV_CODEC_ID_NONE || h > bestHits || (h == bestHits && ev.lastPos > bestPos)) {
            best = id;
            bestPos = ev.lastPos;
            bestHits = h;
        }
    };

    consider(AV_CODEC_ID_HEVC, hevcFinal, hevc);
    consider(AV_CODEC_ID_H264, avcFinal, avc);
    consider(AV_CODEC_ID_MPEG2VIDEO, mpegOk, mpeg2);

    if (best == AV_CODEC_ID_HEVC) {
        dsyslog("vaapivideo/stream: detected HEVC -- mask=0x%02X hits=%d pos=%zu", hevc.seenMask, bestHits,
                hevc.lastPos);
    } else if (best == AV_CODEC_ID_H264) {
        dsyslog("vaapivideo/stream: detected H.264 -- mask=0x%02X hits=%d pos=%zu", avc.seenMask, bestHits,
                avc.lastPos);
    } else if (best == AV_CODEC_ID_MPEG2VIDEO) {
        dsyslog("vaapivideo/stream: detected MPEG-2 -- mask=0x%02X hits=%d pos=%zu", mpeg2.seenMask, bestHits,
                mpeg2.lastPos);
    }

    return best;
}

// ============================================================================
// === SPS PEEK ===
// ============================================================================

namespace {

/// Returns @p nal with emulation-prevention bytes (00 00 03) removed.
/// 0x03 is inserted by the encoder to prevent accidental start-code prefixes inside
/// RBSP payloads; stripping it is required before bit-field parsing (H.264 sec.7.4.1.1,
/// HEVC sec.7.4.2.4).
[[nodiscard]] auto StripEmulationPreventionBytes(std::span<const uint8_t> nal) -> std::vector<uint8_t> {
    std::vector<uint8_t> rbsp;
    rbsp.reserve(nal.size());
    const size_t n = nal.size();
    // NOLINTBEGIN(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- guarded by i+3<n
    for (size_t i = 0; i < n; ++i) {
        // EPB is only valid followed by {0x00..0x03} (H.264 sec.7.4.1.1). Checking the next
        // byte disambiguates a real EPB from `00 00 03 <other>` in malformed input.
        if (i + 3 < n && nal[i] == 0x00 && nal[i + 1] == 0x00 && nal[i + 2] == 0x03 && nal[i + 3] <= 0x03) {
            rbsp.push_back(0x00);
            rbsp.push_back(0x00);
            i += 2; // skip the 0x03 stuffing byte
        } else {
            rbsp.push_back(nal[i]);
        }
    }
    // NOLINTEND(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
    return rbsp;
}

/// Bounds-checked bitstream reader for SPS parsing. Overrun returns 0 and sets
/// the @c overran flag; callers check Overran() before committing parsed values.
class BitReader {
  public:
    explicit BitReader(std::span<const uint8_t> data) noexcept : data_{data} {}

    /// Read @p n bits (n in [0, 32]); returns 0 and sets overran on overrun.
    [[nodiscard]] auto ReadBits(int n) noexcept -> uint32_t {
        uint32_t value = 0;
        for (int i = 0; i < n; ++i) {
            if (bitPos_ >= data_.size() * 8) [[unlikely]] {
                overran_ = true;
                return 0;
            }
            const size_t byteIdx = bitPos_ / 8;
            const int bitInByte = 7 - static_cast<int>(bitPos_ % 8);
            // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- bitPos_ just checked
            value = (value << 1) | ((data_[byteIdx] >> bitInByte) & 0x01U);
            ++bitPos_;
        }
        return value;
    }

    /// Unsigned Exp-Golomb (H.264 sec.9.1 / HEVC sec.9.2). Leading-zero count capped at 32
    /// to prevent runaway on malformed or adversarial input.
    [[nodiscard]] auto ReadUe() noexcept -> uint32_t {
        int leadingZeros = 0;
        while (leadingZeros < 32 && ReadBits(1) == 0 && !overran_) {
            ++leadingZeros;
        }
        if (overran_ || leadingZeros >= 32) [[unlikely]] {
            overran_ = true;
            return 0;
        }
        const uint32_t suffix = ReadBits(leadingZeros);
        return (1U << leadingZeros) - 1U + suffix;
    }

    auto Skip(int n) noexcept -> void {
        const auto step = static_cast<size_t>(n);
        if (bitPos_ + step > data_.size() * 8) [[unlikely]] {
            overran_ = true;
            return;
        }
        bitPos_ += step;
    }

    [[nodiscard]] auto Overran() const noexcept -> bool { return overran_; }

  private:
    std::span<const uint8_t> data_;
    size_t bitPos_{0};
    bool overran_{false};
};

/// Returns a span over the NAL payload (after the header) for the first Annex-B
/// NAL matching @p targetType, or an empty span if not found.
/// H.264: type = low 5 bits of header byte 0; header = 1 byte.
/// HEVC:  type = bits [1..6] of header byte 0 (H.265 sec.7.3.1.2); header = 2 bytes.
[[nodiscard]] auto FindNal(std::span<const uint8_t> data, AVCodecID codec, uint8_t targetType) noexcept
    -> std::span<const uint8_t> {
    const size_t size = data.size();
    if (size < 4) {
        return {};
    }
    const uint8_t *p = data.data();

    for (size_t i = 0; i + 4 <= size;) {
        const auto *found = static_cast<const uint8_t *>(std::memchr(p + i, 0x00, size - i));
        if (!found) {
            break;
        }
        i = static_cast<size_t>(found - p);

        const size_t startCodeLen = AnnexBStartCodeLength(p, size, i);
        if (startCodeLen == 0) {
            ++i;
            continue;
        }
        const size_t nalPos = i + startCodeLen;
        if (nalPos >= size) {
            break;
        }

        uint8_t nalType = 0;
        size_t headerLen = 0;
        if (codec == AV_CODEC_ID_H264) {
            // H.264 NAL header: 1 byte. Reject forbidden_zero_bit != 0.
            if ((p[nalPos] & 0x80) != 0) {
                i = nalPos;
                continue;
            }
            nalType = p[nalPos] & 0x1F;
            headerLen = 1;
        } else if (codec == AV_CODEC_ID_HEVC) {
            // HEVC NAL header: 2 bytes. Reject forbidden_zero_bit != 0.
            if (nalPos + 1 >= size || (p[nalPos] & 0x80) != 0) {
                i = nalPos;
                continue;
            }
            nalType = (p[nalPos] >> 1) & 0x3F;
            headerLen = 2;
        } else {
            return {};
        }

        if (nalType == targetType) {
            const size_t payloadStart = nalPos + headerLen;
            if (payloadStart >= size) {
                return {};
            }
            // NAL ends at the next Annex-B start code (00 00 01 or 00 00 00 01), not at
            // EPB sequences 00 00 03 which are interior to the NAL payload.
            size_t end = size;
            for (size_t j = payloadStart; j + 2 < size; ++j) {
                if (p[j] != 0x00 || p[j + 1] != 0x00) {
                    continue;
                }
                if (p[j + 2] == 0x01 || (j + 3 < size && p[j + 2] == 0x00 && p[j + 3] == 0x01)) {
                    end = j;
                    break;
                }
            }
            return data.subspan(payloadStart, end - payloadStart);
        }

        i = nalPos;
    }
    return {};
}

/// True iff @p profileIdc carries the extended SPS fields (chroma_format_idc,
/// bit_depth_luma_minus8, etc. from H.264 sec.7.3.2.1.1). This is NOT the same as
/// "high-bit-depth": High(100) and Scalable Baseline(83) are 8-bit but still carry
/// the syntax, so the cursor must traverse these fields to reach any that follow.
[[nodiscard]] constexpr auto HasH264ExtendedSpsFields(int profileIdc) noexcept -> bool {
    switch (profileIdc) {
        case 44:  // CAVLC 4:4:4 Intra
        case 83:  // Scalable Baseline
        case 86:  // Scalable High
        case 100: // High
        case 110: // High 10
        case 118: // Multiview High
        case 122: // High 4:2:2
        case 128: // Stereo High
        case 134: // MFC High
        case 135: // MFC Depth High
        case 138: // Multiview Depth High
        case 139: // Enhanced Multiview Depth High
        case 244: // High 4:4:4 Predictive
            return true;
        default:
            return false;
    }
}

/// Parse H.264 SPS for profile / level / bit-depth / chroma. Extended fields
/// (chroma_format_idc, bit_depth_luma_minus8) are only present in profiles
/// that HasH264ExtendedSpsFields() returns true for.
[[nodiscard]] auto ProbeH264Sps(std::span<const uint8_t> sps) noexcept -> VideoStreamInfo {
    VideoStreamInfo info;
    info.codecId = AV_CODEC_ID_H264;
    if (sps.size() < 3) {
        return info;
    }

    // profile_idc | constraint_set_flags | level_idc are the first three byte-aligned fields.
    // NOLINTBEGIN(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- size >= 3 checked above
    info.profile = static_cast<int>(sps[0]);
    info.level = static_cast<int>(sps[2]);
    // NOLINTEND(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)

    const std::vector<uint8_t> rbsp = StripEmulationPreventionBytes(sps.subspan(3));
    BitReader br{std::span<const uint8_t>{rbsp.data(), rbsp.size()}};

    (void)br.ReadUe(); // seq_parameter_set_id
    if (br.Overran()) [[unlikely]] {
        return info;
    }

    if (HasH264ExtendedSpsFields(info.profile)) {
        const uint32_t chromaFormatIdc = br.ReadUe();
        if (chromaFormatIdc == 3) {
            (void)br.ReadBits(1); // separate_colour_plane_flag
        }
        const uint32_t bitDepthLumaMinus8 = br.ReadUe();
        if (br.Overran()) {
            return info;
        }
        if (chromaFormatIdc <= 3) {
            info.chroma = static_cast<ChromaFormat>(chromaFormatIdc > 0 ? chromaFormatIdc - 1 : 0);
        }
        info.bitDepth = (bitDepthLumaMinus8 >= 2) ? BitDepth::k10 : BitDepth::k8;
        info.hasSps = true;
    } else {
        // Baseline/Main profiles are 8-bit 4:2:0 by spec; no extension fields to parse.
        info.bitDepth = BitDepth::k8;
        info.chroma = ChromaFormat::k420;
        info.hasSps = true;
    }

    return info;
}

/// Parse HEVC SPS for profile / level / bit-depth / chroma / coded dimensions.
/// Walks profile_tier_level (H.265 sec.7.3.3) then seq_parameter_set_rbsp (sec.7.3.2.2.1).
[[nodiscard]] auto ProbeHevcSps(std::span<const uint8_t> sps) noexcept -> VideoStreamInfo {
    VideoStreamInfo info;
    info.codecId = AV_CODEC_ID_HEVC;
    if (sps.size() < 2) {
        return info;
    }
    const std::vector<uint8_t> rbsp = StripEmulationPreventionBytes(sps);
    BitReader br{std::span<const uint8_t>{rbsp.data(), rbsp.size()}};

    br.Skip(4); // sps_video_parameter_set_id
    const uint32_t maxSubLayersMinus1 = br.ReadBits(3);
    br.Skip(1); // sps_temporal_id_nesting_flag

    // profile_tier_level(profilePresentFlag=1, maxNumSubLayersMinus1):
    br.Skip(2);                                      // general_profile_space
    (void)br.ReadBits(1);                            // general_tier_flag
    info.profile = static_cast<int>(br.ReadBits(5)); // general_profile_idc
    br.Skip(32);                                     // general_profile_compatibility_flag[0..31]
    br.Skip(48);                                     // general_* constraint flags (4 + 43 + 1 bits)
    info.level = static_cast<int>(br.ReadBits(8));   // general_level_idc

    std::array<bool, 8> subProfilePresent{};
    std::array<bool, 8> subLevelPresent{};
    // HEVC spec caps maxSubLayersMinus1 at 6; clamp to 8 defensively for malformed input.
    // NOLINTBEGIN(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- subLayerLimit <= 8
    const uint32_t subLayerLimit = std::min<uint32_t>(maxSubLayersMinus1, 8U);
    for (uint32_t i = 0; i < subLayerLimit; ++i) {
        subProfilePresent[i] = br.ReadBits(1) != 0;
        subLevelPresent[i] = br.ReadBits(1) != 0;
    }
    if (maxSubLayersMinus1 > 0) {
        for (uint32_t i = maxSubLayersMinus1; i < 8; ++i) {
            br.Skip(2); // reserved_zero_2bits
        }
    }
    for (uint32_t i = 0; i < subLayerLimit; ++i) {
        if (subProfilePresent[i]) {
            br.Skip(2 + 1 + 5 + 32 + 48); // profile_space + tier + profile_idc + compat_flags + constraints
        }
        if (subLevelPresent[i]) {
            br.Skip(8); // sub_layer_level_idc
        }
    }
    // NOLINTEND(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)

    // seq_parameter_set_rbsp() fields after profile_tier_level():
    (void)br.ReadUe(); // sps_seq_parameter_set_id
    const uint32_t chromaFormatIdc = br.ReadUe();
    if (chromaFormatIdc == 3) {
        br.Skip(1); // separate_colour_plane_flag
    }
    const uint32_t picWidth = br.ReadUe();  // pic_width_in_luma_samples
    const uint32_t picHeight = br.ReadUe(); // pic_height_in_luma_samples
    if (br.ReadBits(1) != 0) {              // conformance_window_flag
        (void)br.ReadUe();                  // conf_win_left_offset
        (void)br.ReadUe();                  // conf_win_right_offset
        (void)br.ReadUe();                  // conf_win_top_offset
        (void)br.ReadUe();                  // conf_win_bottom_offset
    }
    const uint32_t bitDepthLumaMinus8 = br.ReadUe();

    if (br.Overran()) {
        return info;
    }

    if (chromaFormatIdc <= 3) {
        info.chroma = static_cast<ChromaFormat>(chromaFormatIdc > 0 ? chromaFormatIdc - 1 : 0);
    }
    info.codedWidth = static_cast<int>(picWidth);
    info.codedHeight = static_cast<int>(picHeight);
    info.bitDepth = (bitDepthLumaMinus8 >= 2) ? BitDepth::k10 : BitDepth::k8;
    info.hasSps = true;
    return info;
}

/// Parse MPEG-2 sequence_header (ISO 13818-2 sec.6.2.2.1) for coded dimensions.
/// Only horizontal/vertical size are extracted; profile and level live in the
/// optional sequence_extension (0xB5) and are not needed for DVB main-profile streams.
[[nodiscard]] auto ProbeMpeg2SequenceHeader(std::span<const uint8_t> data) noexcept -> VideoStreamInfo {
    VideoStreamInfo info;
    info.codecId = AV_CODEC_ID_MPEG2VIDEO;

    const size_t size = data.size();
    if (size < 7) {
        return info;
    }
    const uint8_t *p = data.data();

    for (size_t i = 0; i + 7 <= size;) {
        const auto *found = static_cast<const uint8_t *>(std::memchr(p + i, 0x00, size - i));
        if (!found) {
            break;
        }
        i = static_cast<size_t>(found - p);

        const size_t startCodeLen = AnnexBStartCodeLength(p, size, i);
        if (startCodeLen == 0) {
            ++i;
            continue;
        }
        const size_t headerPos = i + startCodeLen;
        if (headerPos + 4 > size) {
            break;
        }
        // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic) -- guarded by headerPos+4 <= size
        if (p[headerPos] != 0xB3) { // not sequence_header_code; skip
            i = headerPos;
            continue;
        }
        const uint32_t hv = AV_RB24(p + headerPos + 1);
        info.codedWidth = static_cast<int>((hv >> 12) & 0x0FFF);
        info.codedHeight = static_cast<int>(hv & 0x0FFF);
        // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        info.hasSps = info.codedWidth > 0 && info.codedHeight > 0;
        return info;
    }
    return info;
}

} // namespace

auto ProbeVideoSps(AVCodecID codec, std::span<const uint8_t> accessUnit) noexcept -> VideoStreamInfo {
    VideoStreamInfo info;
    info.codecId = codec;

    if (codec == AV_CODEC_ID_H264) {
        const auto sps = FindNal(accessUnit, codec, 7); // NAL type 7 = SPS
        if (!sps.empty()) {
            return ProbeH264Sps(sps);
        }
    } else if (codec == AV_CODEC_ID_HEVC) {
        const auto sps = FindNal(accessUnit, codec, 33); // NAL type 33 = SPS
        if (!sps.empty()) {
            return ProbeHevcSps(sps);
        }
    } else if (codec == AV_CODEC_ID_MPEG2VIDEO) {
        // Populates codedWidth/codedHeight for right-size surface pre-allocation;
        // profile/level are not parsed (DVB main-profile is always 8-bit 4:2:0).
        return ProbeMpeg2SequenceHeader(accessUnit);
    }
    // AV1/unknown: DVB does not carry AV1 in-band; the mediaplayer path supplies
    // profile/bit-depth from AVCodecParameters. Return hasSps=false so the caller
    // opens an 8-bit surface and reopens on format mismatch.
    return info;
}
