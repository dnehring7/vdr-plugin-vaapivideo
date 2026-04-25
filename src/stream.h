// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Dirk Nehring <dnehring@gmx.net>
/**
 * @file stream.h
 * @brief Shared stream data model: codec detection, SPS probe, backend-selection
 *        tables, and the IMediaSource seam for a future mediaplayer path.
 *
 * Single home for "what is this elementary stream?" Both the VDR PES path and a
 * future FFmpeg libavformat-based mediaplayer path populate VideoStreamInfo /
 * AudioStreamInfo from this header.
 *
 * Not here: PES container framing (pes.h), DRM/VA-API details (display.h, decoder.h).
 */

#ifndef VDR_VAAPIVIDEO_STREAM_H
#define VDR_VAAPIVIDEO_STREAM_H

#include "caps.h"
#include "common.h"

// ============================================================================
// === STREAM DATA MODEL ===
// ============================================================================

/// Luma bit depth from SPS / sequence header. Drives backend-selection (HEVC Main 10
/// vs Main, etc.). The filter chain emits P010 only when HDR passthrough is active;
/// 10-bit SDR streams are downconverted to NV12. To route 10-bit SDR to P010 surfaces,
/// update the call site in filter.cpp.
enum class BitDepth : uint8_t {
    k8 = 0,
    k10 = 1,
};

/// Chroma sub-sampling. VA driver and DRM plane layout support only 4:2:0 today;
/// 4:2:2 / 4:4:4 are reserved for future SW-fallback codecs in the mediaplayer path.
enum class ChromaFormat : uint8_t {
    k420 = 0,
    k422 = 1,
    k444 = 2,
};

/// Video elementary stream descriptor: everything the decoder needs to select a
/// backend and configure the filter chain. Populated by ProbeVideoSps() on the
/// first access unit (PES path) or from AVCodecParameters (mediaplayer path).
/// Numeric fields default to "unknown/unspecified"; hasSps distinguishes a
/// partial guess from a validated parse.
struct VideoStreamInfo {
    BitDepth bitDepth{BitDepth::k8};                               ///< Luma bit-depth (drives backend selection)
    ChromaFormat chroma{ChromaFormat::k420};                       ///< Chroma sub-sampling
    AVCodecID codecId{AV_CODEC_ID_NONE};                           ///< FFmpeg codec id
    int codedHeight{0};                                            ///< Coded picture height (luma samples)
    int codedWidth{0};                                             ///< Coded picture width (luma samples)
    AVColorPrimaries primaries{AVCOL_PRI_UNSPECIFIED};             ///< VUI colour primaries
    AVColorRange range{AVCOL_RANGE_UNSPECIFIED};                   ///< VUI colour range (Limited / Full)
    AVColorSpace colorSpace{AVCOL_SPC_UNSPECIFIED};                ///< VUI matrix coefficients
    AVColorTransferCharacteristic transfer{AVCOL_TRC_UNSPECIFIED}; ///< VUI transfer function
    const uint8_t *extradata{nullptr}; ///< Non-owning init data (SPS/PPS/VPS concat); nullptr on PES path
    int extradataSize{0};              ///< Byte length of @c extradata
    bool hasSps{false};                ///< True iff ProbeVideoSps parsed an authoritative in-band parameter set
    int level{0};                      ///< Codec level (level_idc or general_level_idc)
    int profile{AV_PROFILE_UNKNOWN};   ///< e.g. AV_PROFILE_HEVC_MAIN_10, AV_PROFILE_AV1_MAIN
};

/// Audio elementary stream descriptor. Field order is preserved from the former
/// AudioStreamParams so designated-initialiser call sites in audio.cpp compile unchanged.
struct AudioStreamInfo {
    int channels{0};                     ///< Decoded output channel count
    AVCodecID codecId{AV_CODEC_ID_NONE}; ///< FFmpeg codec identifier
    const uint8_t *extradata{nullptr};   ///< Non-owning; caller owns the buffer lifetime
    int extradataSize{0};                ///< Byte length of @c extradata
    int sampleRate{0};                   ///< Audio sample rate in Hz
};

// ============================================================================
// === BACKEND SELECTION TABLES ===
// ============================================================================

/// One row in the backend selection table: (codec, profile, bit-depth) -> GpuCaps flag.
/// profile == AV_PROFILE_UNKNOWN is a wildcard (matches any profile). First matching
/// row in kVideoBackendTable wins; rows must be ordered most-specific first.
/// Adding a codec: one row here + one case in ProbeGpuCaps.
struct VideoBackendCap {
    AVCodecID codecId;
    int profile; ///< AV_PROFILE_UNKNOWN = match-all
    BitDepth bitDepth;
    bool GpuCaps::*flag;
};

inline constexpr std::array<VideoBackendCap, 7> kVideoBackendTable{{
    {.codecId = AV_CODEC_ID_MPEG2VIDEO,
     .profile = AV_PROFILE_UNKNOWN,
     .bitDepth = BitDepth::k8,
     .flag = &GpuCaps::hwMpeg2},
    {.codecId = AV_CODEC_ID_H264,
     .profile = AV_PROFILE_H264_HIGH_10,
     .bitDepth = BitDepth::k10,
     .flag = &GpuCaps::hwH264High10},
    {.codecId = AV_CODEC_ID_H264, .profile = AV_PROFILE_UNKNOWN, .bitDepth = BitDepth::k8, .flag = &GpuCaps::hwH264},
    {.codecId = AV_CODEC_ID_HEVC,
     .profile = AV_PROFILE_HEVC_MAIN_10,
     .bitDepth = BitDepth::k10,
     .flag = &GpuCaps::hwHevcMain10},
    {.codecId = AV_CODEC_ID_HEVC, .profile = AV_PROFILE_UNKNOWN, .bitDepth = BitDepth::k8, .flag = &GpuCaps::hwHevc},
    {.codecId = AV_CODEC_ID_AV1,
     .profile = AV_PROFILE_AV1_MAIN,
     .bitDepth = BitDepth::k10,
     .flag = &GpuCaps::hwAv1Main10},
    {.codecId = AV_CODEC_ID_AV1, .profile = AV_PROFILE_UNKNOWN, .bitDepth = BitDepth::k8, .flag = &GpuCaps::hwAv1},
}};

/// Look up the GpuCaps member-pointer that gates hardware decode for @p info.
/// Dereference with caps.*(result). Returns nullptr if no table row matches.
[[nodiscard]] auto SelectVideoBackendCap(const VideoStreamInfo &info) noexcept -> bool GpuCaps::*;

/// Codecs the S/PDIF muxer can IEC61937-wrap. Sink availability (ELD check)
/// is separate; use AudioSinkCaps::Supports (caps.h) for that.
inline constexpr std::array<AVCodecID, 6> kAudioPassthroughTable{{
    AV_CODEC_ID_AC3,
    AV_CODEC_ID_EAC3,
    AV_CODEC_ID_DTS,
    AV_CODEC_ID_TRUEHD,
    AV_CODEC_ID_AC4,
    AV_CODEC_ID_MPEGH_3D_AUDIO,
}};

/// True iff @p codec has an IEC61937 wrapper (table lookup only; no sink ELD check).
[[nodiscard]] auto IsPassthroughCapable(AVCodecID codec) noexcept -> bool;

// A null member-pointer in any row would silently make SelectVideoBackendCap return
// nullptr for every query, disabling hardware decode without any runtime error.
static_assert(std::ranges::all_of(kVideoBackendTable,
                                  [](const VideoBackendCap &c) constexpr noexcept -> bool {
                                      return c.flag != nullptr;
                                  }),
              "kVideoBackendTable has a row with a null GpuCaps flag pointer");

// ============================================================================
// === CODEC DETECTION ===
// ============================================================================

/// Identify an audio codec from ES bytes (AC-3/E-AC-3/AAC/AAC-LATM/DTS/MP2/TrueHD).
/// Linear sync-word scan; first decisive match wins. Returns AV_CODEC_ID_NONE if
/// nothing is recognised.
[[nodiscard]] auto DetectAudioCodec(std::span<const uint8_t> data) noexcept -> AVCodecID;

/// Identify a video codec from ES bytes (HEVC/H.264/MPEG-2). Weighted multi-codec
/// Annex-B scan with cross-codec phantom-hit invalidation. Returns AV_CODEC_ID_NONE
/// when evidence is insufficient.
[[nodiscard]] auto DetectVideoCodec(std::span<const uint8_t> data) noexcept -> AVCodecID;

/// Minimal in-band parameter-set peek. Per-codec coverage:
///   H.264  (NAL type 7):  profile / level / bit-depth / chroma
///   HEVC   (NAL type 33): profile / level / bit-depth / chroma / coded size
///   MPEG-2 (start 0xB3):  coded size only
/// VUI colour metadata is not parsed. AV1 is not parsed (DVB does not carry it
/// in-band; the mediaplayer path gets profile/bit-depth from AVCodecParameters).
/// Returns hasSps=false on error or unsupported codec; callers should wait for
/// an access unit with parameter sets rather than committing to an 8-bit guess.
[[nodiscard]] auto ProbeVideoSps(AVCodecID codec, std::span<const uint8_t> accessUnit) noexcept -> VideoStreamInfo;

// ============================================================================
// === MEDIAPLAYER SEAM ===
// ============================================================================

/// Abstract input source for the decoder pipeline. Currently unused (no implementations
/// or callers ship yet); declared here so a future cFfmpegMediaSource over libavformat
/// can be added without reshuffling this header. The PES path in device.cpp continues
/// to push raw bytes directly into cVaapiDecoder::EnqueueData / cAudioProcessor::Decode.
///
/// Contract:
///  - AVPacket::pts is in the 90 kHz VDR domain, already rebased from the source timebase.
///  - AVPacket buffers are owned by the packet; consumer calls av_packet_unref.
///  - Read*Packet may be called concurrently; implementations own the locking policy.
class IMediaSource {
  public:
    IMediaSource() = default;
    virtual ~IMediaSource() noexcept = default;
    IMediaSource(const IMediaSource &) = delete;
    IMediaSource(IMediaSource &&) noexcept = delete;
    auto operator=(const IMediaSource &) -> IMediaSource & = delete;
    auto operator=(IMediaSource &&) noexcept -> IMediaSource & = delete;

    /// Pull one video packet. Returns 0 (success), AVERROR(EAGAIN) (not ready),
    /// or AVERROR_EOF. @p out is populated by av_packet_ref; caller unrefs.
    [[nodiscard]] virtual auto ReadVideoPacket(AVPacket *out) -> int = 0;

    /// Pull one audio packet; same semantics as ReadVideoPacket.
    [[nodiscard]] virtual auto ReadAudioPacket(AVPacket *out) -> int = 0;

    /// Video stream descriptor. Stable after Open(); undefined before.
    [[nodiscard]] virtual auto VideoInfo() const noexcept -> const VideoStreamInfo & = 0;

    /// Audio stream descriptor. Stable after Open(); undefined before.
    [[nodiscard]] virtual auto AudioInfo() const noexcept -> const AudioStreamInfo & = 0;

    /// Drop buffered data (seek, channel switch, teardown). Safe to call concurrently
    /// with Read*Packet(); implementations must cause those to return AVERROR(EAGAIN)
    /// as soon as the flush takes effect.
    virtual auto Flush() -> void = 0;
};

#endif // VDR_VAAPIVIDEO_STREAM_H
