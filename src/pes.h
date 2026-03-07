// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Dirk Nehring <dnehring@gmx.net>
/**
 * @file pes.h
 * @brief PES packet parsing and codec detection
 */

#ifndef VDR_VAAPIVIDEO_PES_H
#define VDR_VAAPIVIDEO_PES_H

#include "common.h"

// ============================================================================
// === PES PACKET ===
// ============================================================================

/// Non-owning decoded view of a PES packet produced by ParsePes().
struct PesPacket {
    bool isAudio{};              ///< Audio stream (0xC0-0xDF or private 0xBD)
    bool isVideo{};              ///< Video stream (0xE0-0xEF)
    const uint8_t *payload{};    ///< ES payload following PES header
    size_t payloadSize{};        ///< ES payload byte count
    int64_t dts{AV_NOPTS_VALUE}; ///< Decode timestamp (90 kHz)
    int64_t pts{AV_NOPTS_VALUE}; ///< Presentation timestamp (90 kHz)
};

// ============================================================================
// === PUBLIC API ===
// ============================================================================

[[nodiscard]] auto DetectAudioCodec(std::span<const uint8_t> data) noexcept
    -> AVCodecID; ///< Identify audio codec from ES bytes (AC-3/E-AC-3/AAC/DTS/MP2)
[[nodiscard]] auto DetectVideoCodec(std::span<const uint8_t> data) noexcept
    -> AVCodecID; ///< Identify video codec from ES bytes (HEVC/H.264/MPEG-2)
[[nodiscard]] auto ParsePes(std::span<const uint8_t> data) noexcept -> PesPacket; ///< Parse PES packet: extract stream
                                                                                  ///< type, timestamps and payload span

#endif // VDR_VAAPIVIDEO_PES_H
