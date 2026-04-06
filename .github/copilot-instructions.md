# VDR VAAPI Video Plugin -- Coding Guidelines

## Architecture

```
VDR --PES--> cVaapiDevice --> [Decoder, Audio, Display] --> Hardware (zero-copy VAAPI -> DRM)
```

| File | Role |
|------|------|
| `common.h` | `AvErr()` helper, RAII deleters, version guards, shared headers |
| `config.cpp` | Plugin configuration, display parameters, setup storage |
| `decoder.cpp` | VAAPI decode, filter graphs, A/V sync |
| `device.cpp` | VDR integration, PES routing, lifecycle |
| `display.cpp` | DRM atomic modesetting, page flips |
| `audio.cpp` | ALSA output, IEC61937 passthrough |
| `osd.cpp` | Hardware OSD overlay |
| `pes.cpp` | PES parsing, codec detection |

## Code Style

### Signatures

Trailing return types on **all** functions (including `-> void`, deleted operators).
`[[nodiscard]]` on every value-returning function.
`noexcept` on destructors, move operations, and trivial `const` getters.

```cpp
[[nodiscard]] auto Initialize(std::string_view device) const -> bool;
auto Clear() -> void;
auto operator=(const T&) -> T& = delete;
auto operator=(T&&) noexcept -> T&;
~Component() noexcept;
[[nodiscard]] auto GetValue() const noexcept -> int;
```

### Expressions

- `auto` when the type is evident from the initializer.
- `std::format` for string construction (never `snprintf` for `std::string`).
- `std::span` for buffer parameters (never raw pointer + size).
- `std::memcpy` / `std::memset` (always `std::`-qualified).
- Brace initialization for members; designated initializers for structs.
- Structured bindings for multi-value returns: `const auto [ptr, ec] = std::from_chars(...)`.

```cpp
const auto width = static_cast<uint32_t>(rawWidth);
auto name = std::format("hw:{}", cardId);
auto ParsePes(std::span<const uint8_t> data) -> PesPacket;
StreamParams params{.codecId = AV_CODEC_ID_H264, .sampleRate = 48000};
```

### Control Flow

- `[[unlikely]]` on error paths; `[[likely]]` on hot paths.
- Early returns to avoid deep nesting.

```cpp
if (!ptr) [[unlikely]] { return false; }
if (hasData) [[likely]] { process(); }
```

## File Organization

### Include Guards

`#ifndef` / `#define` / `#endif` (never `#pragma once`):

```cpp
#ifndef VDR_VAAPIVIDEO_MODULE_H
#define VDR_VAAPIVIDEO_MODULE_H
// ...
#endif // VDR_VAAPIVIDEO_MODULE_H
```

### Section Headers

Full-width ruler with centered label:

```cpp
// ============================================================================
// === CONSTANTS ===
// ============================================================================
```

### Declarations

- `inline constexpr` in headers; plain `constexpr` in `.cpp` files. UPPERCASE names:

```cpp
// header
inline constexpr int SHUTDOWN_TIMEOUT_MS = 5000; ///< Thread shutdown timeout (ms)
// cpp
constexpr size_t DECODER_QUEUE_CAPACITY = 500;   ///< Video packet queue depth
```

- `static` for file-local free functions; anonymous namespace for file-local types:

```cpp
[[nodiscard]] static auto Helper(int x) -> int;
namespace { struct Internal { ... }; } // namespace
```

### Documentation

Inline Doxygen after the member. Alphabetical within logical groups.

```cpp
AVFrame* frame{};             ///< Brief description
int64_t pts{AV_NOPTS_VALUE};  ///< Presentation timestamp (90 kHz)
```

## Logging

Format: `"vaapivideo/<component>: <message>"` where `<component>` is the source file basename.
Never log in hot paths (decode/render loops) -- errors and initialization only.

```cpp
isyslog("vaapivideo/decoder: initialized");              // Info
dsyslog("vaapivideo/decoder: packet queue size=%zu", n); // Debug
esyslog("vaapivideo/decoder: failed: %s", msg);          // Error
```

## Resource Management

### VDR Lifecycle

```cpp
auto Clear() -> void;     ///< Reset buffers, keep resources
auto Shutdown() -> void;  ///< Release all resources, log stats
~Component() noexcept;    ///< Call Shutdown() if not already done
```

### RAII Deleters (common.h)

All FFmpeg/DRM resources wrapped in `std::unique_ptr` with custom deleters.
Drain queues via RAII -- never manual `av_packet_free`:

```cpp
std::unique_ptr<AVFrame, FreeAVFrame> frame{av_frame_alloc()};
std::unique_ptr<AVPacket, FreeAVPacket> pkt{av_packet_alloc()};
std::unique_ptr<AVCodecContext, FreeAVCodecContext> ctx{avcodec_alloc_context3(codec)};

while (!packetQueue.empty()) {
    const std::unique_ptr<AVPacket, FreeAVPacket> dropped{packetQueue.front()};
    packetQueue.pop();
}
```

## Concurrency

### VDR Primitives Only

Never use `std::thread`, `std::mutex`, etc.

| Primitive | Usage |
|-----------|-------|
| `cThread` | Inherit, override `Action()` |
| `cMutex` / `cMutexLock` | Always declare lock as `const` |
| `cCondVar` | Condition variable, paired with `cMutex` |
| `cCondWait::SleepMs()` | Timed sleep utility |
| `cTimeMs` | Deadline timer, always `const` |

### Thread Safety

VDR's `Running()` is NOT thread-safe -- use atomic flags instead:

```cpp
std::atomic<bool> stopping{};   ///< Thread stop request
std::atomic<bool> hasExited{};  ///< Thread exit confirmation
```

Action() loop:

```cpp
while (!stopping.load(std::memory_order_acquire)) { /* work */ }
hasExited.store(true, std::memory_order_release);  // BEFORE final log
```

Shutdown() pattern:

```cpp
auto Shutdown() -> void {
    if (stopping.exchange(true, std::memory_order_acq_rel)) return;
    condition.Broadcast();
    if (!hasExited.load(std::memory_order_acquire)) Cancel(3);
    const cTimeMs timeout(SHUTDOWN_TIMEOUT_MS);
    while (!hasExited.load(std::memory_order_acquire) && !timeout.TimedOut()) {
        condition.Broadcast();
        cCondWait::SleepMs(10);
    }
}
```

## Error Handling

FFmpeg errors use `AvErr()` from `common.h`. DRM/VAAPI errors use `strerror(errno)` or `%m`:

```cpp
if (const int ret = avcodec_send_packet(ctx, pkt); ret < 0) {
    esyslog("vaapivideo/decoder: send_packet failed: %s", AvErr(ret).data());
    return false;
}
if (drmModeAddFB2(fd, w, h, fmt, handles, pitches, offsets, &fbId, 0) != 0) {
    esyslog("vaapivideo/display: AddFB2 failed: %m");
    return false;
}
```

## Suppression Policy

`NOLINT` only at C API boundaries (FFmpeg, VAAPI, DRM, ALSA). Always specify the check name:

```cpp
reinterpret_cast<const AVDRMFrameDescriptor*>(...) // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
uint8_t* outPtr = buf.data();                      // NOLINT(misc-const-correctness)
```

Prefer `std::memcpy` over `reinterpret_cast` when possible.

## Rules

**Required:** `[[nodiscard]]` on all value-returning functions -- trailing return types on all functions -- `const` correctness on functions, parameters, and locks -- RAII for all resources (no naked `new`/`delete`) -- check all C API return values -- `std::`-qualified C functions.

**Forbidden:** exceptions (VDR is exception-free) -- `std::thread`/`std::mutex` (use VDR primitives) -- `#pragma once` -- static mutable state -- legacy APIs (require FFmpeg 7.0+, VDR 2.6.0+).
