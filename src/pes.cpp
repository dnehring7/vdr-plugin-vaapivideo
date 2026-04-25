// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Dirk Nehring <dnehring@gmx.net>
/**
 * @file pes.cpp
 * @brief PES packet parsing (ISO 13818-1 sec.2.4.3.6).
 *
 * ParsePes() extracts stream-id flags, PTS/DTS, and a payload span from one PES
 * packet. Returns a zero-initialized result on any error rather than reading past
 * the supplied buffer.
 *
 * Codec detection previously lived here but was moved to stream.cpp where it can
 * share the elementary-stream byte inspection logic with the SPS peek.
 */

#include "pes.h"

// C++ Standard Library
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>

// FFmpeg
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wsign-conversion"
extern "C" {
#include <libavutil/avutil.h>
#include <libavutil/intreadwrite.h>
}
#pragma GCC diagnostic pop

// ============================================================================
// === CONSTANTS ===
// ============================================================================

// Byte offsets and sizes within a PES packet header (ISO 13818-1 sec.2.4.3.6/7).
constexpr size_t PES_HEADER_MIN_SIZE = 6U;     ///< prefix(3) + stream_id(1) + PES_packet_length(2)
constexpr size_t PES_OFFSET_FLAGS2 = 7U;       ///< Flags byte 2; PTS_DTS_flags in bits 7--6
constexpr size_t PES_OFFSET_HDR_DATA_LEN = 8U; ///< PES_header_data_length field
constexpr size_t PES_HEADER_EXT_OFFSET = 9U;   ///< First byte of the variable-length header extension
constexpr size_t PES_TIMESTAMP_SIZE = 5U;      ///< 33-bit PTS/DTS on the wire: 3+1+2+1+2+1 bytes with marker bits

constexpr uint32_t PES_START_CODE_PREFIX = 0x000001U; ///< 24-bit start_code_prefix (sec.2.4.3.6)

// stream_id ranges (sec.2.4.3.7). DVB carries AC-3/DTS/AAC in private_stream_1 (0xBD),
// so it is treated as audio alongside the standard MPEG audio range 0xC0--0xDF.
constexpr uint8_t PES_STREAM_ID_AUDIO_FIRST = 0xC0; ///< MPEG audio base; mask 0xE0 covers 0xC0--0xDF
constexpr uint8_t PES_STREAM_ID_PRIVATE = 0xBD;     ///< private_stream_1 (DVB AC-3/DTS/AAC)
constexpr uint8_t PES_STREAM_ID_VIDEO_FIRST = 0xE0; ///< MPEG video base; mask 0xF0 covers 0xE0--0xEF

// ============================================================================
// === INTERNAL HELPERS ===
// ============================================================================

namespace {

[[nodiscard]] inline auto ParseTimestamp(const uint8_t *bytes) noexcept -> int64_t {
    // 33-bit value packed as three fragments (3+15+15 bits), each word ending with a
    // mandatory marker bit (sec.2.4.3.7). All three marker bits must be 1; a missing marker
    // means the data is corrupt. Rejecting here prevents garbage timestamps from reaching
    // the A/V sync controller.
    if ((AV_RB8(bytes) & AV_RB8(bytes + 2) & AV_RB8(bytes + 4) & 0x01) == 0) [[unlikely]] {
        return AV_NOPTS_VALUE;
    }

    const uint64_t b32_30 = (AV_RB8(bytes) >> 1) & 0x07;
    const uint64_t b29_15 = (AV_RB16(bytes + 1) >> 1) & 0x7FFF;
    const uint64_t b14_0 = (AV_RB16(bytes + 3) >> 1) & 0x7FFF;

    return static_cast<int64_t>((b32_30 << 30) | (b29_15 << 15) | b14_0);
}

} // namespace

// ============================================================================
// === PES PARSING ===
// ============================================================================

auto ParsePes(std::span<const uint8_t> data) noexcept -> PesPacket {
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

    // PTS_DTS_flags (bits 7--6 of flags2): 0x80 = PTS only, 0xC0 = PTS+DTS, 0x00 = none.
    // 0x40 is illegal per spec. Only the two legal "has PTS" patterns are handled;
    // everything else leaves pts/dts as AV_NOPTS_VALUE.
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

    // PES_packet_length (bytes 4--5): 0 means "unbounded", legal only on video in TS.
    // VDR invariant: cTsToPes delivers a complete reassembled packet, so pesLen==0 safely
    // maps to the full buffer remainder. Non-zero values are clamped to the actual buffer
    // size as a defensive measure against truncation.
    // Note: PES_packet_length counts from byte 6 onward and does NOT include the 6-byte
    // fixed header, so the payload starts at byte (6 + headerExtLen) = headerSize.
    size_t payloadLen = size - headerSize;
    const uint16_t pesLen = AV_RB16(p + 4);

    if (pesLen > 0) {
        const auto declared = static_cast<size_t>(pesLen);
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
