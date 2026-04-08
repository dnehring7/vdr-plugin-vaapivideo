// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Dirk Nehring <dnehring@gmx.net>
/**
 * @file pes.cpp
 * @brief PES packet parsing and codec detection
 */

#include "pes.h"

// C++ Standard Library
#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>

// FFmpeg
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wsign-conversion"
extern "C" {
#include <libavcodec/codec_id.h>
#include <libavutil/avutil.h>
#include <libavutil/intreadwrite.h>
}
#pragma GCC diagnostic pop

// VDR
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wvariadic-macros"
#include <vdr/tools.h>
#pragma GCC diagnostic pop

// ============================================================================
// === CONSTANTS ===
// ============================================================================

constexpr size_t PES_HEADER_EXT_OFFSET = 9U;   ///< Extension data offset (after flags + header_data_length)
constexpr size_t PES_HEADER_MIN_SIZE = 6U;     ///< Minimum PES: prefix(3) + stream_id(1) + length(2)
constexpr size_t PES_OFFSET_FLAGS2 = 7U;       ///< Second flags byte (PTS_DTS_flags in bits 7-6)
constexpr size_t PES_OFFSET_HDR_DATA_LEN = 8U; ///< PES_header_data_length field index
constexpr size_t PES_TIMESTAMP_SIZE = 5U;      ///< Bytes per PTS/DTS timestamp field

constexpr uint32_t PES_START_CODE_PREFIX = 0x000001U; ///< 24-bit start code prefix (ISO 13818-1)

constexpr uint8_t PES_STREAM_ID_AUDIO_FIRST = 0xC0; ///< First MPEG audio stream_id (0xC0-0xDF)
constexpr uint8_t PES_STREAM_ID_PRIVATE = 0xBD;     ///< Private stream 1 (AC-3/DTS/AAC)
constexpr uint8_t PES_STREAM_ID_VIDEO_FIRST = 0xE0; ///< First MPEG video stream_id (0xE0-0xEF)

// ============================================================================
// === INTERNAL HELPERS ===
// ============================================================================

[[nodiscard]] static inline auto ParseTimestamp(const uint8_t *bytes) noexcept -> int64_t {
    // Decode 33-bit PTS/DTS from 5 PES header bytes (ISO 13818-1 sec.2.4.3.7).
    // Each of the 3 timestamp fragments ends with a marker bit that must be 1; reject corrupt timestamps early.
    if ((AV_RB8(bytes) & AV_RB8(bytes + 2) & AV_RB8(bytes + 4) & 0x01) == 0) [[unlikely]] {
        return AV_NOPTS_VALUE;
    }

    const uint64_t b32_30 = (AV_RB8(bytes) >> 1) & 0x07;
    const uint64_t b29_15 = (AV_RB16(bytes + 1) >> 1) & 0x7FFF;
    const uint64_t b14_0 = (AV_RB16(bytes + 3) >> 1) & 0x7FFF;

    return static_cast<int64_t>((b32_30 << 30) | (b29_15 << 15) | b14_0);
}

/// Per-codec evidence accumulator: each strong NAL type maps to a bit in seenMask.
namespace {
struct CodecEvidence {
    uint8_t seenMask{}; ///< Bit flags of distinct strong NAL types seen
    size_t lastPos{};   ///< Buffer offset of most recent evidence

    auto Record(uint8_t bit, size_t pos) noexcept -> void {
        seenMask |= bit;
        lastPos = pos;
    }
};
} // namespace

// ============================================================================
// === CODEC DETECTION ===
// ============================================================================

[[nodiscard]] auto DetectAudioCodec(std::span<const uint8_t> data) noexcept -> AVCodecID {
    // Single-pass scan for audio sync words; first match wins (codecs have unambiguous sync patterns).
    if (data.size() < 4) [[unlikely]] {
        return AV_CODEC_ID_NONE;
    }

    const size_t size = data.size();
    const uint8_t *p = data.data();

    for (size_t i = 0; i + 4 <= size; ++i) {
        const uint16_t sync = AV_RB16(p + i);

        // AAC ADTS must be tested before MP2: both share 0xFFE prefix, but ADTS has layer=00 (0xFFF6==0xFFF0)
        // which MP2's looser 0xFFE0 mask would also match.
        if ((sync & 0xFF00) == 0xFF00) [[unlikely]] {
            if ((sync & 0xFFF6) == 0xFFF0) [[unlikely]] {
                return AV_CODEC_ID_AAC;
            }
            if ((sync & 0xFFE0) == 0xFFE0 && (sync & 0x06) != 0x00) [[likely]] {
                return AV_CODEC_ID_MP2;
            }
        }

        // AAC-LATM / LOAS (ISO 14496-3): 11-bit sync 0x2B7 in upper bits of 0x56E0 mask.
        // Unlike other codecs, LATM's short sync triggers false positives; require a second sync at the
        // predicted next frame boundary (current + 3 header bytes + audioMuxLengthBytes).
        if ((sync & 0xFFE0) == 0x56E0) [[unlikely]] {
            const auto frameLen = static_cast<uint16_t>(((sync & 0x1FU) << 8) | AV_RB8(p + i + 2));
            if (frameLen >= 2) {
                const size_t next = i + 3 + frameLen;
                if (next + 2 <= size && (AV_RB16(p + next) & 0xFFE0) == 0x56E0) {
                    return AV_CODEC_ID_AAC_LATM;
                }
            }
        }

        // AC-3/E-AC-3 sync 0x0B77; bsid > 10 -> E-AC-3 (ATSC A/52).
        if (sync == 0x0B77) [[unlikely]] {
            if (i + 5 < size && ((AV_RB8(p + i + 5) >> 3) & 0x1F) > 10) [[unlikely]] {
                return AV_CODEC_ID_EAC3;
            }
            return AV_CODEC_ID_AC3;
        }

        // DTS core sync 0x7FFE8001 (ETSI TS 102 114).
        if (AV_RB32(p + i) == 0x7FFE8001) [[unlikely]] {
            return AV_CODEC_ID_DTS;
        }
    }

    return AV_CODEC_ID_NONE;
}

[[nodiscard]] auto DetectVideoCodec(std::span<const uint8_t> data) noexcept -> AVCodecID {
    // Multi-codec scan: accumulate per-codec evidence from Annex-B NAL units until one codec reaches threshold.
    // Thresholds: HEVC: VPS (unique) or 2+ param sets. H.264: 2+ of SPS/PPS/IDR. MPEG-2: sequence_header + ext/GOP.
    // Key challenge: MPEG-2 start codes (0xB3/0xB5/0xB8) alias as valid H.264/HEVC NAL headers; confirmed
    // MPEG-2 therefore invalidates H.264/HEVC evidence.
    const size_t size = data.size();
    if (size < 6) [[unlikely]] {
        return AV_CODEC_ID_NONE;
    }

    const uint8_t *p = data.data();

    CodecEvidence hevc{};
    CodecEvidence avc{};
    CodecEvidence mpeg2{};

    // HEVC evidence bits: 0x01=VPS(32), 0x02=SPS(33), 0x04=PPS(34), 0x08=IDR/CRA
    constexpr uint8_t kHevcParamMask = 0x07;

    for (size_t i = 0; i + 4 <= size;) {
        // Fast-skip non-zero bytes.
        const auto *found = static_cast<const uint8_t *>(std::memchr(p + i, 0x00, size - i));
        if (!found) [[unlikely]] {
            break;
        }
        i = static_cast<size_t>(found - p);

        if (i + 4 > size) [[unlikely]] {
            break;
        }

        if (p[i + 1] != 0x00) {
            ++i;
            continue;
        }

        // Detect start code: 00 00 01 (3-byte) or 00 00 00 01 (4-byte).
        size_t scLen = 0;
        if (p[i + 2] == 0x01) {
            scLen = 3;
        } else if (p[i + 2] == 0x00 && p[i + 3] == 0x01) {
            scLen = 4;
        }

        if (scLen == 0) {
            ++i;
            continue;
        }

        const size_t nalPos = i + scLen;
        if (nalPos >= size) [[unlikely]] {
            break;
        }

        const uint8_t b0 = p[nalPos];

        // MPEG-2 start codes 0xB3/0xB5/0xB8 are unambiguous: b0 >= 0x80 means forbidden_zero_bit==1 which is
        // illegal for both H.264 and HEVC, so these cannot be NAL headers from either codec.
        // Evidence bits: sequence_header(0xB3)=0x01, extension(0xB5)=0x02, GOP(0xB8)=0x04.
        if (b0 == 0xB3 || b0 == 0xB5 || b0 == 0xB8) {
            uint8_t bit = 0x04;
            if (b0 == 0xB3) {
                bit = 0x01;
            } else if (b0 == 0xB5) {
                bit = 0x02;
            }
            mpeg2.Record(bit, i);
            if ((mpeg2.seenMask & 0x01) != 0 && (mpeg2.seenMask & 0x06) != 0) [[unlikely]] {
                break; // sequence_header + extension/GOP -> definitive MPEG-2
            }
        }

        // HEVC NAL header: forbidden_zero_bit(1) nal_unit_type(6) nuh_layer_id_msb(1) | [byte1].
        // Require nuh_layer_id==0 for VPS/SPS/PPS: rejects H.264 slice NALs that alias as HEVC param sets
        // (e.g. H.264 nal_unit_type=1 with nal_ref_idc=2 -> byte 0x41 -> HEVC type 32 = VPS).
        if (nalPos + 1 < size && (b0 & 0x80) == 0) {
            const uint8_t b1 = p[nalPos + 1];
            const uint8_t temporalIdPlus1 = b1 & 0x07;

            if (temporalIdPlus1 != 0) {
                const uint8_t hevcType = (b0 >> 1) & 0x3F;
                const bool layerIdZero = ((b0 & 0x01) == 0) && (((b1 >> 3) & 0x1F) == 0);
                uint8_t bit = 0;

                if (hevcType >= 32 && hevcType <= 34 && layerIdZero) {
                    bit = static_cast<uint8_t>(1U << (hevcType - 32)); // VPS=0x01, SPS=0x02, PPS=0x04
                } else if (hevcType >= 19 && hevcType <= 21 && layerIdZero) {
                    // IDR_W_RADL(19)/IDR_N_LP(20)/CRA(21): only count after a param set is seen to avoid
                    // premature confirmation from H.264 aliases (H.264 0x27->type 19, 0x28->type 20).
                    if ((hevc.seenMask & kHevcParamMask) != 0) {
                        bit = 0x08;
                    }
                }

                if (bit != 0) {
                    hevc.Record(bit, i);
                }
            }
        }

        // H.264 NAL header: forbidden_zero_bit(1) nal_ref_idc(2) nal_unit_type(5).
        // nal_ref_idc > 0 is mandatory for SPS(7)/PPS(8)/IDR(5) per H.264 sec.7.4.1 and conveniently
        // eliminates MPEG-2 start codes 0x01-0x1F which would decode as nal_ref_idc==0.
        if ((b0 & 0x80) == 0 && (b0 & 0x60) != 0) {
            const uint8_t avcType = b0 & 0x1F;
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

    // HEVC: require VPS (unique to HEVC) or 2+ distinct param set types.
    const int hevcHits = std::popcount(hevc.seenMask);
    const int hevcParams = std::popcount(static_cast<uint8_t>(hevc.seenMask & kHevcParamMask));
    const bool hevcOk = (hevcHits >= 2) && ((hevc.seenMask & 0x01) != 0 || hevcParams >= 2);

    const bool avcOk = std::popcount(avc.seenMask) >= 2;
    const bool mpegOk = ((mpeg2.seenMask & 0x01) != 0) && ((mpeg2.seenMask & 0x06) != 0);

    // MPEG-2 confirmation voids H.264/HEVC: its start codes (0xB3/0xB5/0xB8) are valid NAL byte patterns
    // that inflate H.264/HEVC evidence counts with phantom hits.
    const bool hevcFinal = hevcOk && !mpegOk;
    const bool avcFinal = avcOk && !mpegOk;

    // Prefer more evidence (hit count); break ties by latest-in-buffer position.
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
        dsyslog("vaapivideo/pes: detected HEVC -- mask=0x%02X hits=%d pos=%zu", hevc.seenMask, bestHits, hevc.lastPos);
    } else if (best == AV_CODEC_ID_H264) {
        dsyslog("vaapivideo/pes: detected H.264 -- mask=0x%02X hits=%d pos=%zu", avc.seenMask, bestHits, avc.lastPos);
    } else if (best == AV_CODEC_ID_MPEG2VIDEO) {
        dsyslog("vaapivideo/pes: detected MPEG-2 -- mask=0x%02X hits=%d pos=%zu", mpeg2.seenMask, bestHits,
                mpeg2.lastPos);
    }

    return best;
}

// ============================================================================
// === PES PARSING ===
// ============================================================================

[[nodiscard]] auto ParsePes(std::span<const uint8_t> data) noexcept -> PesPacket {
    PesPacket result{};

    const size_t size = data.size();
    if (size < PES_HEADER_MIN_SIZE) [[unlikely]] {
        return result;
    }

    const uint8_t *p = data.data();

    if (AV_RB24(p) != PES_START_CODE_PREFIX) [[unlikely]] {
        return result;
    }

    const uint8_t streamId = p[3];

    // Video 0xE0-0xEF, audio 0xC0-0xDF, private_stream_1 0xBD.
    const bool isVideo = (streamId & 0xF0) == PES_STREAM_ID_VIDEO_FIRST;
    const bool isAudio = ((streamId & 0xE0) == PES_STREAM_ID_AUDIO_FIRST) || (streamId == PES_STREAM_ID_PRIVATE);

    if (!isVideo && !isAudio) [[unlikely]] {
        return result;
    }

    result.isVideo = isVideo;
    result.isAudio = isAudio;

    if (size < PES_HEADER_EXT_OFFSET) [[unlikely]] {
        return result;
    }

    const uint8_t headerExtLen = p[PES_OFFSET_HDR_DATA_LEN];
    const size_t headerSize = PES_HEADER_EXT_OFFSET + headerExtLen;

    if (headerSize > size) [[unlikely]] {
        return result;
    }

    // PTS_DTS_flags: 0x80 = PTS only, 0xC0 = PTS + DTS.
    const uint8_t ptsDtsFlags = p[PES_OFFSET_FLAGS2] & 0xC0;

    if (ptsDtsFlags == 0x80) [[likely]] {
        if (headerExtLen >= PES_TIMESTAMP_SIZE) [[likely]] {
            result.pts = ParseTimestamp(p + PES_HEADER_EXT_OFFSET);
        }
    } else if (ptsDtsFlags == 0xC0) [[unlikely]] {
        if (headerExtLen >= PES_TIMESTAMP_SIZE * 2) [[likely]] {
            result.pts = ParseTimestamp(p + PES_HEADER_EXT_OFFSET);
            result.dts = ParseTimestamp(p + PES_HEADER_EXT_OFFSET + PES_TIMESTAMP_SIZE);
        }
    }

    // PES_packet_length (bytes 4-5): 0 means unbounded (ISO 13818-1 permits this only for video streams
    // carried in Transport Stream packets; VDR always passes complete PES packets, so 0 is safe to treat as "use all
    // remaining data").
    size_t payloadLen = size - headerSize;
    const uint16_t pesLen = AV_RB16(p + 4);

    if (pesLen > 0) {
        const auto declared = static_cast<size_t>(pesLen);
        // PES_packet_length counts from byte 6 (after start_code_prefix + stream_id + length field itself).
        if (declared + 6 > headerSize) {
            payloadLen = std::min(declared + 6 - headerSize, payloadLen);
        } else {
            payloadLen = 0;
        }
    }

    if (payloadLen > 0) [[likely]] {
        result.payload = p + headerSize;
        result.payloadSize = payloadLen;
    }

    return result;
}
