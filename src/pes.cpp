// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Dirk Nehring <dnehring@gmx.net>
/**
 * @file pes.cpp
 * @brief PES header parsing + bitstream-based codec detection.
 *
 * Two unrelated jobs sharing a file because both consume PES bytes:
 *   - ParsePes(): split a PES packet into stream-id flags + PTS/DTS + payload span.
 *     Tolerant of truncation (returns an empty result rather than reading past end).
 *   - Detect{Audio,Video}Codec(): scan the payload for codec-specific sync patterns
 *     (audio: single sync word) or NAL-unit evidence (video: weighted multi-codec).
 *
 * Both functions are pure / noexcept / called from the receiver thread on every
 * packet of a fresh stream until the device pins a codec id. They MUST be cheap;
 * the video detector uses memchr() to skip over zero-free runs and stops at the
 * first decisive evidence threshold.
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

// All offsets are into the PES header (ISO 13818-1 sec.2.4.3.6).
constexpr size_t PES_HEADER_EXT_OFFSET = 9U;   ///< First byte after the fixed header (= start of extension)
constexpr size_t PES_HEADER_MIN_SIZE = 6U;     ///< prefix(3) + stream_id(1) + PES_packet_length(2)
constexpr size_t PES_OFFSET_FLAGS2 = 7U;       ///< Second flags byte; PTS_DTS_flags in bits 7..6
constexpr size_t PES_OFFSET_HDR_DATA_LEN = 8U; ///< PES_header_data_length (size of the extension area)
constexpr size_t PES_TIMESTAMP_SIZE = 5U;      ///< Wire size of one PTS or DTS field (33 bits + markers)

constexpr uint32_t PES_START_CODE_PREFIX = 0x000001U; ///< 24-bit start_code_prefix (ISO 13818-1 sec.2.4.3.6)

// stream_id ranges defined by ISO 13818-1 sec.2.4.3.7. Audio is matched as
// 0xC0..0xDF *or* private_stream_1 (0xBD), the latter is how DVB carries AC-3 / DTS / etc.
constexpr uint8_t PES_STREAM_ID_AUDIO_FIRST = 0xC0; ///< MPEG audio stream_id base (range 0xC0..0xDF)
constexpr uint8_t PES_STREAM_ID_PRIVATE = 0xBD;     ///< private_stream_1 (DVB AC-3 / DTS / AAC carriage)
constexpr uint8_t PES_STREAM_ID_VIDEO_FIRST = 0xE0; ///< MPEG video stream_id base (range 0xE0..0xEF)

// ============================================================================
// === INTERNAL HELPERS ===
// ============================================================================

[[nodiscard]] static inline auto ParseTimestamp(const uint8_t *bytes) noexcept -> int64_t {
    // Decode a 33-bit PTS/DTS field (ISO 13818-1 sec.2.4.3.7). The 33 bits are split into
    // three fragments (3+15+15) that each end with a marker bit that the spec requires to
    // be 1. AND'ing the marker bytes with 0x01 catches corrupt timestamps cheaply -- we
    // return NOPTS rather than feeding garbage timestamps into the A/V sync controller.
    if ((AV_RB8(bytes) & AV_RB8(bytes + 2) & AV_RB8(bytes + 4) & 0x01) == 0) [[unlikely]] {
        return AV_NOPTS_VALUE;
    }

    const uint64_t b32_30 = (AV_RB8(bytes) >> 1) & 0x07;
    const uint64_t b29_15 = (AV_RB16(bytes + 1) >> 1) & 0x7FFF;
    const uint64_t b14_0 = (AV_RB16(bytes + 3) >> 1) & 0x7FFF;

    return static_cast<int64_t>((b32_30 << 30) | (b29_15 << 15) | b14_0);
}

/// Tracks accumulated codec evidence during a DetectVideoCodec() scan. Each "strong"
/// NAL/start-code type maps to one bit; lastPos lets us tie-break codecs with the same
/// hit count by preferring evidence later in the buffer (the more-recent decoder state
/// is more reliable than what's left in the receiver ring buffer from the prior channel).
namespace {
struct CodecEvidence {
    uint8_t seenMask{}; ///< Distinct strong-NAL bits seen so far
    size_t lastPos{};   ///< Buffer offset of the most recent matching NAL (for tie-break)

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
    // Linear sync-word scan, first decisive match wins. The order of tests matters --
    // see the AAC-vs-MP2 and AAC-LATM hazards inline below.
    if (data.size() < 4) [[unlikely]] {
        return AV_CODEC_ID_NONE;
    }

    const size_t size = data.size();
    const uint8_t *p = data.data();

    for (size_t i = 0; i + 4 <= size; ++i) {
        const uint16_t sync = AV_RB16(p + i);

        // ORDER MATTERS: AAC ADTS uses layer==00 (sync 0xFFF1/0xFFF9, masked 0xFFF6==0xFFF0).
        // MP2 frame sync uses layer==01/10/11 with the looser 0xFFE0 mask which would
        // *also* match every ADTS frame if tested first. Test ADTS first to disambiguate.
        if ((sync & 0xFF00) == 0xFF00) [[unlikely]] {
            if ((sync & 0xFFF6) == 0xFFF0) [[unlikely]] {
                return AV_CODEC_ID_AAC;
            }
            // (sync & 0x06) != 0 == "layer field is not the reserved value 00" -- this is
            // the MPEG audio frame header validity bit that excludes random 0xFFE? bytes.
            if ((sync & 0xFFE0) == 0xFFE0 && (sync & 0x06) != 0x00) [[likely]] {
                return AV_CODEC_ID_MP2;
            }
        }

        // AAC-LATM / LOAS (ISO 14496-3 sec.1.7.3) syncword is only 11 bits (0x2B7), much
        // shorter than the others, so single-frame matches are unreliable. Confirm with a
        // second sync at the predicted next frame: current pos + 3-byte header + the
        // audioMuxLengthBytes field embedded in the lower 13 bits of the header.
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
        // byte+5 bits 7..3, ATSC A/52 sec.A.4.3):
        //   0..10  -> AC-3 (8 is the typical real-world value)
        //   11..15 -> reserved by spec; we classify as E-AC-3 to match FFmpeg, but no
        //             real stream uses these so the choice is academic
        //   16     -> E-AC-3
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
    }

    return AV_CODEC_ID_NONE;
}

[[nodiscard]] auto DetectVideoCodec(std::span<const uint8_t> data) noexcept -> AVCodecID {
    // Walk Annex-B start codes once and accumulate per-codec "strong NAL" evidence into
    // three CodecEvidence accumulators. Final thresholds (computed at the bottom):
    //   HEVC    -- popcount(seenMask) >= 2 AND (VPS bit set OR popcount(params) >= 2)
    //   H.264   -- popcount(seenMask) >= 2 of {SPS, PPS, IDR}
    //   MPEG-2  -- sequence_header (0xB3) AND (extension 0xB5 OR GOP 0xB8)
    //
    // Cross-codec hazard: MPEG-2 start codes 0xB3/0xB5/0xB8 are valid byte patterns when
    // interpreted as H.264 / HEVC NAL headers and would inflate avc/hevc evidence with
    // phantom hits. We compensate by INVALIDATING avc/hevc evidence when MPEG-2 reaches
    // its threshold (the !mpegOk gates below).
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
        // memchr-skip the long zero-free runs that dominate compressed video. Faster than
        // a per-byte loop on every architecture libc supports (SIMD-accelerated on x86_64).
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

        // Annex-B start code: either 00 00 01 (3 bytes) or 00 00 00 01 (4 bytes).
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

        // MPEG-2 sequence_header / extension / GOP. Bytes 0xB3/0xB5/0xB8 all have the
        // high bit set, which would set forbidden_zero_bit=1 in an H.264/HEVC NAL header
        // -- both spec-illegal -- so these patterns are unambiguous for MPEG-2.
        // mpeg2 seenMask: sequence_header(0xB3)=0x01, extension(0xB5)=0x02, GOP(0xB8)=0x04.
        if (b0 == 0xB3 || b0 == 0xB5 || b0 == 0xB8) {
            uint8_t bit = 0x04;
            if (b0 == 0xB3) {
                bit = 0x01;
            } else if (b0 == 0xB5) {
                bit = 0x02;
            }
            mpeg2.Record(bit, i);
            // Early exit: sequence_header + (extension OR GOP) is the same threshold the
            // bottom-of-function check uses. Skip the remaining scan once we have it.
            if ((mpeg2.seenMask & 0x01) != 0 && (mpeg2.seenMask & 0x06) != 0) [[unlikely]] {
                break;
            }
        }

        // HEVC NAL header (2 bytes, ITU-T H.265 sec.7.3.1.2):
        //   b0 = forbidden_zero_bit(1) | nal_unit_type(6) | nuh_layer_id_high(1)
        //   b1 = nuh_layer_id_low(5)   | nuh_temporal_id_plus1(3)
        // Require nuh_layer_id==0 for VPS/SPS/PPS to reject H.264 slice NALs that alias
        // these byte patterns (e.g. H.264 byte 0x41 = nal_ref_idc=2 type=1 also decodes
        // as HEVC type 32 = VPS, but its byte 1 layer bits are essentially random and
        // rarely all zero). nuh_temporal_id_plus1 must also be != 0 (the spec encodes
        // temporal_id+1, so 0 is reserved/illegal) -- another cheap H.264 reject.
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
                    // IDR_W_RADL(19)/IDR_N_LP(20)/CRA(21): only count AFTER any param set
                    // has already been seen. H.264 byte 0x27 / 0x28 alias as HEVC types
                    // 19 / 20, so an IDR-only match is not enough to call HEVC.
                    if ((hevc.seenMask & kHevcParamMask) != 0) {
                        bit = 0x08;
                    }
                }

                if (bit != 0) {
                    hevc.Record(bit, i);
                }
            }
        }

        // H.264 NAL header (1 byte, H.264 sec.7.3.1):
        //   forbidden_zero_bit(1) | nal_ref_idc(2) | nal_unit_type(5)
        // The spec requires nal_ref_idc != 0 for SPS(7) / PPS(8) / IDR(5) (sec.7.4.1).
        // Bonus: that mask also rejects MPEG-2 start codes 0x01..0x1F (which would have
        // nal_ref_idc==0 if reinterpreted), so we don't accidentally count them here.
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

    // HEVC threshold: at least 2 distinct strong NALs total, AND either VPS (essentially
    // unique to HEVC because H.264 has no equivalent type) or 2+ distinct param set types.
    // The "VPS or 2+ params" disjunction lets a stream with VPS+IDR confirm (real HEVC)
    // while still requiring richer evidence when VPS is missing.
    const int hevcHits = std::popcount(hevc.seenMask);
    const int hevcParams = std::popcount(static_cast<uint8_t>(hevc.seenMask & kHevcParamMask));
    const bool hevcOk = (hevcHits >= 2) && ((hevc.seenMask & 0x01) != 0 || hevcParams >= 2);

    const bool avcOk = std::popcount(avc.seenMask) >= 2;
    const bool mpegOk = ((mpeg2.seenMask & 0x01) != 0) && ((mpeg2.seenMask & 0x06) != 0);

    // Apply the cross-codec invalidation documented at the top of the function: confirmed
    // MPEG-2 voids any avc/hevc evidence (those bits are phantom hits from MPEG-2 start
    // codes interpreted as NAL headers).
    const bool hevcFinal = hevcOk && !mpegOk;
    const bool avcFinal = avcOk && !mpegOk;

    // Tie-break: more distinct evidence wins; same hit count -> later in buffer wins
    // (the more recent decoder state is more likely to reflect the *current* stream
    // and not stale bytes left in the receiver ring from the previous channel).
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

    // Range matching:
    //   video 0xE0..0xEF -> mask 0xF0
    //   audio 0xC0..0xDF -> mask 0xE0 (covers both 0xC0..0xC7 and 0xC8..0xDF in one test)
    //   private_stream_1 0xBD is the DVB transport for AC-3/DTS/etc. and is treated as audio.
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

    // PTS_DTS_flags (top 2 bits of flags2): 0x80 = PTS only, 0xC0 = PTS + DTS.
    // 0x40 is reserved/illegal per spec; 0x00 = no timestamps. We only act on the two
    // legal "has PTS" forms; everything else leaves result.pts as AV_NOPTS_VALUE.
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

    // PES_packet_length (bytes 4..5):
    //   spec sentinel: 0 means "unbounded", legal only on video streams in TS containers.
    //   VDR contract:  cTsToPes always hands us a complete PES packet, so for unbounded
    //                  length we can safely consume the rest of the input buffer.
    //   our handling:  unbounded -> use buffer remainder; non-zero -> trust declared
    //                  length but clamp to what's actually in the buffer (defensive).
    // CAUTION: a non-VDR caller that passes truncated PES bytes with pesLen==0 would
    // get whatever junk follows in the buffer here. If this function is ever reused
    // outside the vdr-plugin-vaapivideo PES path, the unbounded branch needs review.
    size_t payloadLen = size - headerSize;
    const uint16_t pesLen = AV_RB16(p + 4);

    if (pesLen > 0) {
        const auto declared = static_cast<size_t>(pesLen);
        // PES_packet_length is measured from byte 6 onward (i.e. it does NOT include
        // the 6-byte fixed header that contains the length field itself).
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
