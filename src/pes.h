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
