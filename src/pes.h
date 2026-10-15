// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Dirk Nehring <dnehring@gmx.net>
/**
 * @file pes.h
 * @brief PES packet parsing (ISO 13818-1 sec.2.4.3.6).
 *
 * Codec detection lives in stream.h -- it operates on elementary-stream bytes,
 * not on the PES container.
 */

#ifndef VDR_VAAPIVIDEO_PES_H
#define VDR_VAAPIVIDEO_PES_H

#include "common.h"

// ============================================================================
// === PES PACKET ===
// ============================================================================

/// private_stream_1 (ISO 13818-1): DVB carries AC-3/E-AC-3/DTS/AAC in it. Shared with the device, which keys
/// the dolby track switch on it.
inline constexpr uint8_t PES_STREAM_ID_PRIVATE = 0xBD;

/// DVD private_stream_1 audio payloads start with a substream header (id, frame count, first access-unit
/// pointer) the decoder must not see. DVB has none.
inline constexpr size_t PES_DVD_SUBSTREAM_HEADER_BYTES = 4;

/// The codec a DVD audio substream id names: AC-3 0x80-0x87, DTS 0x88-0x8F, LPCM 0xA0-0xA7. On DVD the id is
/// the detection, not the frames: LPCM has no sync word, and a 2 KB pack mostly holds one header of the AC-3 or
/// DTS frames it splits, which frame detection can't corroborate -- 5.1 AC-3 at 448 kbit/s (1792-byte frames)
/// never gave two decisive packs in a row, so its re-detection after any in-session reset never confirmed.
[[nodiscard]] constexpr auto DvdSubstreamCodec(uint8_t id) noexcept -> AVCodecID {
    switch (id & 0xF8) {
        case 0x80:
            return AV_CODEC_ID_AC3;
        case 0x88:
            return AV_CODEC_ID_DTS;
        case 0xA0:
            return AV_CODEC_ID_PCM_DVD;
        default:
            return AV_CODEC_ID_NONE;
    }
}

/// Non-owning view into the input buffer passed to ParsePes(); lifetime is bounded by that buffer.
struct PesPacket {
    bool isAudio{};              ///< stream_id 0xC0--0xDF or private_stream_1 (0xBD, DVB AC-3/DTS)
    bool isVideo{};              ///< stream_id 0xE0--0xEF
    const uint8_t *payload{};    ///< Points into the original input span; null if packet has no payload
    size_t payloadSize{};        ///< ES payload byte count (0 when payload is null)
    int64_t dts{AV_NOPTS_VALUE}; ///< Decode timestamp, 90 kHz; AV_NOPTS_VALUE if absent or corrupt
    int64_t pts{AV_NOPTS_VALUE}; ///< Presentation timestamp, 90 kHz; AV_NOPTS_VALUE if absent or corrupt
};

// ============================================================================
// === PUBLIC API ===
// ============================================================================

/// Parse one PES packet from @p data. Returns a zero-initialized PesPacket on any error
/// (truncated input, bad start code, unsupported stream_id). Never reads past @p data.end().
[[nodiscard]] auto ParsePes(std::span<const uint8_t> data) noexcept -> PesPacket;

#endif // VDR_VAAPIVIDEO_PES_H
