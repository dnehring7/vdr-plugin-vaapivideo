# Coding Rules

README.md = architecture / build. `src/common.h` = RAII deleters + `AvErr()`.

Build: `make` | Lint: `make lint` (bear + clang-tidy) | Format: `make indent`.

## Hard rules

- Trailing return type on every function (incl. `-> void`, deleted ops).
- `[[nodiscard]]` on every value-returning function.
- `noexcept` on dtors, moves, trivial const getters.
- Include guards `#ifndef/#define/#endif` -- no `#pragma once`.
- `std::format` (never `snprintf` into `std::string`). `std::span` (never `ptr+size`). `std::`-qualified C funcs.
- RAII for all resources; no naked `new`/`delete`, no manual `av_*_free`.
- Check every C-API return value. No exceptions (VDR is exception-free).
- VDR threading primitives only: `cThread`, `cMutex`, `cMutexLock`, `cCondVar`, `cCondWait::SleepMs()`, `cTimeMs` -- no `std::thread`/`std::mutex`.
- `NOLINT` only at C-API boundaries; always name the check.

## Conventions

- Constants: `inline constexpr` UPPERCASE in headers, plain `constexpr` in `.cpp`.
- File scope: `static` for free fns, anon namespace for types.
- Logging: `"vaapivideo/<basename>: <msg>"`; never in hot paths.
- Errors: FFmpeg -> `AvErr(ret).data()`; DRM/VAAPI -> `%m` or `strerror(errno)`.
- Members: inline Doxygen `///<` after member, brace-init, alphabetical within groups.
- Section rulers: `// === LABEL ===`.
- Hints: `[[unlikely]]` on error paths, `[[likely]]` on hot paths, early returns.

## Thread-safety

VDR's `Running()` is **not** thread-safe. Use `std::atomic<bool> stopping` / `hasExited` with `acquire`/`release`. Mirror existing `Shutdown()` patterns.

## Lifecycle (match existing patterns)

- `Clear()` -- reset buffers, keep resources.
- `Shutdown()` -- release everything, log stats, idempotent via `stopping.exchange(true)`.
- `~Destructor() noexcept` -- calls `Shutdown()` if not yet done.
