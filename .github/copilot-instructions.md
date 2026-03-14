# VDR VAAPI Video Plugin -- Coding Guidelines


## Architecture

```
VDR --PES--> cVaapiDevice --> [Decoder, Audio, Display] --> Hardware (zero-copy VAAPI -> DRM)
```

- `audio.cpp` -- ALSA output, IEC61937 passthrough
- `common.h` -- `AvErr()` helper, RAII deleters, version guards, shared headers
- `config.cpp` -- Plugin configuration, display parameters, setup storage
- `decoder.cpp` -- VAAPI decode, filter graphs, A/V sync
- `device.cpp` -- VDR integration, PES routing, lifecycle
- `display.cpp` -- DRM atomic modesetting, page flips
- `osd.cpp` -- Hardware OSD overlay
- `pes.cpp` -- PES parsing, codec detection


## Code Style

### Signatures

Trailing return types on all functions, including `void` and deleted operators.
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

Use `#ifndef` / `#define` / `#endif` (never `#pragma once`):

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

- `inline constexpr` in headers: `inline constexpr int kTimeout = 500;`
- Plain `constexpr` in `.cpp` files: `constexpr int kTimeout = 500;`
- `static` for file-local free functions (not anonymous namespace): `static auto Helper(int x) -> int;`

### Documentation

Inline Doxygen after the member:

```cpp
AVFrame* frame{};             ///< Brief description
int64_t pts{AV_NOPTS_VALUE};  ///< Presentation timestamp (90 kHz)
```

Ordering: alphabetical within logical groups (constants, public methods, private members, etc.).


## Logging

Format: `"vaapivideo/<component>: <message>"` where `<component>` is the source file basename.

```cpp
isyslog("vaapivideo/decoder: initialized");              // Info
dsyslog("vaapivideo/decoder: packet queue size=%zu", n); // Debug
esyslog("vaapivideo/decoder: failed: %s", msg);          // Error
```

Never log in hot paths (decode/render loops) -- errors and initialization only.


## Resource Management

### VDR Lifecycle

```cpp
auto Clear() -> void;     ///< Reset buffers, keep resources
auto Shutdown() -> void;  ///< Release all resources, log stats
~Component() noexcept;    ///< Call Shutdown() if not already done
```

### RAII Deleters (common.h)

```cpp
std::unique_ptr<AVFrame, FreeAVFrame> frame{av_frame_alloc()};
std::unique_ptr<AVPacket, FreeAVPacket> pkt{av_packet_alloc()};
std::unique_ptr<AVCodecContext, FreeAVCodecContext> ctx{avcodec_alloc_context3(codec)};
```

Drain queues via RAII -- never manual `av_packet_free`:

```cpp
while (!packetQueue.empty()) {
    std::unique_ptr<AVPacket, FreeAVPacket> dropped{packetQueue.front()};
    packetQueue.pop();
}
```


## Concurrency

### VDR Primitives Only

Never use `std::thread`, `std::mutex`, etc.

- `cThread` -- inherit, override `Action()`
- `cMutex` / `cMutexLock` -- always declare lock as `const`
- `cCondVar` -- condition variable, paired with `cMutex`
- `cCondWait::SleepMs()` -- timed sleep utility
- `cTimeMs` -- deadline timer, always `const`

```cpp
const cMutexLock lock(&mutex);
const cTimeMs timeout(500);
```

### Thread Safety

VDR's `Running()` is NOT thread-safe -- use atomic flags instead:

```cpp
std::atomic<bool> stopping{};   ///< Thread stop request
std::atomic<bool> hasExited{};  ///< Thread exit confirmation
```

Action() pattern:

```cpp
while (!stopping.load(std::memory_order_acquire)) {
    // ... work loop with frequent stopping checks
}
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

### FFmpeg Errors

Use `AvErr()` from `common.h` (returns `std::array<char, AV_ERROR_MAX_STRING_SIZE>`):

```cpp
if (const int ret = avcodec_send_packet(ctx, pkt); ret < 0) {
    esyslog("vaapivideo/decoder: send_packet failed: %s", AvErr(ret).data());
    return false;
}
```

### DRM / VAAPI Errors

Log `strerror(errno)` or `%m`:

```cpp
if (drmModeAddFB2(fd, width, height, fmt, handles, pitches, offsets, &fbId, 0) != 0) {
    esyslog("vaapivideo/display: AddFB2 failed: %m");
    return false;
}
```


## Suppression Policy

`NOLINT` is permitted only at C API boundaries (FFmpeg, VAAPI, DRM, ALSA).
Always specify the check name -- never bare `NOLINT`:

```cpp
reinterpret_cast<const AVDRMFrameDescriptor*>(...) // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
uint8_t* outPtr = buf.data();                      // NOLINT(misc-const-correctness)
```

Prefer `std::memcpy` over `reinterpret_cast` when possible (e.g. pixel writes in OSD).


## Quick Reference

Required:

- `[[nodiscard]]` on all value-returning functions
- Trailing return types on all functions (`-> void`, `-> bool`, `-> int`)
- `const` correctness on functions, parameters, and locks
- RAII for all resources -- no naked `new` / `delete`
- Check all C API return values (FFmpeg, DRM, VAAPI, ALSA)
- `std::`-qualified C functions (`std::memcpy`, `std::memset`)

Forbidden:

- Exceptions -- use return values (VDR is exception-free)
- `std::thread`, `std::mutex` -- use VDR primitives (`cThread`, `cMutex`)
- `#pragma once` -- use `#ifndef` / `#define` include guards
- Static mutable state -- use instance members
- Legacy APIs -- require FFmpeg 7.0+, VDR 2.6.0+
