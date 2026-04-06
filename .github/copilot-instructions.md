# Coding Rules

See README.md for architecture, source layout, build instructions, and requirements.
See `src/common.h` for RAII deleters (`FreeAVFrame`, `FreeAVPacket`, …) and `AvErr()`.

## Build & Lint

```sh
make                # build plugin
make lint           # clang-tidy (requires bear + clang-tidy; generates compile_commands.json)
make indent         # clang-format all sources
```

## Hard Rules

- **Trailing return types** on every function, including `-> void` and deleted operators.
- **`[[nodiscard]]`** on every value-returning function.
- **`noexcept`** on destructors, move operations, trivial `const` getters.
- **`#ifndef`/`#define`/`#endif`** include guards — never `#pragma once`.
- **`std::format`** for strings — never `snprintf` into `std::string`.
- **`std::span`** for buffers — never raw pointer + size.
- **`std::`-qualified** C functions (`std::memcpy`, `std::memset`).
- **RAII** for all resources — no naked `new`/`delete`, no manual `av_*_free`.
- **Check every C API return value.**
- **No exceptions** — VDR is exception-free.
- **No `std::thread`/`std::mutex`** — use VDR primitives (`cThread`, `cMutex`, `cMutexLock`, `cCondVar`, `cCondWait::SleepMs()`, `cTimeMs`).
- **`NOLINT`** only at C API boundaries; always name the check.

## Non-Obvious Conventions

- **Constants:** `inline constexpr` + UPPERCASE in headers; plain `constexpr` in `.cpp`.
- **File scope:** `static` for free functions; anonymous namespace for types.
- **Logging:** `"vaapivideo/<basename>: <msg>"` — never in hot paths (decode/render loops).
- **Errors:** FFmpeg → `AvErr(ret).data()`; DRM/VAAPI → `%m` or `strerror(errno)`.
- **Members:** inline Doxygen `///<` after the member, brace-init, alphabetical within groups.
- **Section headers:** full-width `// === LABEL ===` ruler (see existing files).
- **Control flow:** `[[unlikely]]` on error paths, `[[likely]]` on hot paths, early returns.

## Thread-Safety Pitfall

VDR's `Running()` is **not** thread-safe. Use `std::atomic<bool> stopping` / `hasExited` with `memory_order_acquire`/`release`. See existing `Shutdown()` patterns in the codebase.

## Lifecycle

Each component provides three methods — match existing patterns exactly:

- `Clear()` — reset buffers, keep allocated resources.
- `Shutdown()` — release everything, log stats, idempotent via `stopping.exchange(true)`.
- `~Destructor() noexcept` — calls `Shutdown()` if not already done.
