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
    // Decode 33-bit PTS/DTS from 5 PES header bytes (ISO 13818-1 sec.2.4.3.7). Returns AV_NOPTS_VALUE when any marker
    // bit is malformed.
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
    // Scan ES buffer for audio sync words, return first matching codec.
    if (data.size() < 4) [[unlikely]] {
        return AV_CODEC_ID_NONE;
    }

    const size_t size = data.size();
    const uint8_t *p = data.data();

    for (size_t i = 0; i + 4 <= size; ++i) {
        const uint16_t sync = AV_RB16(p + i);

        // AAC ADTS (0xFFF, layer=00) tested before MP2 (0xFFE) -- stricter mask.
        if ((sync & 0xFF00) == 0xFF00) [[unlikely]] {
            if ((sync & 0xFFF6) == 0xFFF0) [[unlikely]] {
                return AV_CODEC_ID_AAC;
            }
            if ((sync & 0xFFE0) == 0xFFE0 && (sync & 0x06) != 0x00) [[likely]] {
                return AV_CODEC_ID_MP2;
            }
        }

        // AAC-LATM / LOAS: 11-bit sync 0x2B7 + 13-bit audioMuxLengthBytes (ISO 14496-3, AudioSyncStream).
        // Require non-zero length to reject false positives from random data (zero-length frames are invalid).
        if ((sync & 0xFFE0) == 0x56E0) [[unlikely]] {
            const auto loasLen = static_cast<uint16_t>(((sync & 0x1FU) << 8) | AV_RB8(p + i + 2));
            if (loasLen > 0) {
                return AV_CODEC_ID_AAC_LATM;
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
    // Evidence-based video codec detection: scan for Annex-B start codes, accumulate per-codec evidence, require 2+
    // distinct strong markers.
    //
    // HEVC: VPS/SPS/PPS (primary, nuh_layer_id==0) + IDR/CRA (secondary). H.264: SPS + PPS + IDR (nal_ref_idc > 0,
    // spec-mandated). MPEG-2: sequence_header + extension/GOP (unambiguous: bit 7 set). AUD excluded (no disambiguation
    // value).
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

        // MPEG-2: sequence_header(0xB3)=0x01, extension(0xB5)=0x02, GOP(0xB8)=0x04.
        {
            uint8_t bit = 0;
            if (b0 == 0xB3) {
                bit = 0x01;
            } else if (b0 == 0xB5) {
                bit = 0x02;
            } else if (b0 == 0xB8) {
                bit = 0x04;
            }
            if (bit != 0) {
                mpeg2.Record(bit, i);
                // Early exit: MPEG-2 markers have bit 7 set (invalid as H.264/HEVC NAL headers). Once sequence_header +
                // extension/GOP found, the result is definitive -- no need to scan further.
                if ((mpeg2.seenMask & 0x01) != 0 && (mpeg2.seenMask & 0x06) != 0) [[unlikely]] {
                    break;
                }
            }
        }

        // HEVC: forbidden_zero_bit(1) nal_unit_type(6) nuh_layer_id_msb(1) | [byte1]. Require nuh_layer_id==0 for param
        // sets (rejects H.264 slice aliases like 0x41->VPS(32), 0x42->SPS(33), 0x44->PPS(34)).
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
                    // IDR/CRA: secondary, only after param set seen. Require layerIdZero to reject H.264 aliases
                    // (0x27->type19, 0x28->type20).
                    if ((hevc.seenMask & kHevcParamMask) != 0) {
                        bit = 0x08;
                    }
                }

                if (bit != 0) {
                    hevc.Record(bit, i);
                }
            }
        }

        // H.264: forbidden_zero_bit(1) nal_ref_idc(2) nal_unit_type(5). Require nal_ref_idc > 0 for SPS/PPS/IDR (spec
        // sec.7.4.1 mandates this; eliminates MPEG-2 slice codes 0x01-0x1F which have nal_ref_idc==0).
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

    // MPEG-2 markers (0xB3/0xB5/0xB8) have bit 7 set, making them invalid as H.264/HEVC NAL headers (forbidden_zero_bit
    // would be 1). When MPEG-2 is confirmed, any H.264/HEVC evidence is from slice-code aliases or stale data.
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

    // PES_packet_length (bytes 4-5): 0 = unspecified (common for video).
    size_t payloadLen = size - headerSize;
    const uint16_t pesLen = AV_RB16(p + 4);

    if (pesLen > 0) {
        const auto declared = static_cast<size_t>(pesLen);
        // declared counts from byte 6 onward; absolute end = declared + 6.
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
