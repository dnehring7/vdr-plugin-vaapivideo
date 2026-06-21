# Coding Rules

README.md = architecture/build; `src/common.h` = RAII deleters + `AvErr()`.
Build `make DEV_WARNINGS=1` (strict `-Werror`; plain `make` is the lenient packaging build) · lint `make lint` (bear+clang-tidy) · format `make indent`.

## Hard rules
- Trailing return type everywhere (incl. `-> void`, deleted ops); `[[nodiscard]]` on every value-returning fn.
- `noexcept` on dtors, moves, trivial const getters.
- Include guards `#ifndef/#define/#endif`, never `#pragma once`.
- `std::format` not `snprintf`+`std::string`; `std::span` not `ptr+size`; `std::`-qualify C funcs.
- RAII-own every resource (no naked `new`/`delete`); direct `av_*_free`/C frees only in deleters, dtors, or cleanup/transfer paths. Check every C-API return.
- No exceptions (VDR is exception-free). `NOLINT` only at C-API boundaries; name the check, give a reason.
- Threads: VDR primitives only (`cThread`, `cMutex`, `cMutexLock`, `cCondVar`, `cCondWait::SleepMs`, `cTimeMs`), never `std::thread`/`std::mutex`. `Running()` is not thread-safe -- gate on `stopping`/`hasExited` atomics (acquire/release).

## Conventions
- Constants: file/namespace scope is `UPPERCASE` (headers `inline constexpr`; `.cpp` `constexpr` in the anonymous namespace); function-local `constexpr` is `kCamelCase`.
- File-local helpers (free fns, types): anonymous namespace, co-located with use (multiple per file ok); no new file-scope `static`. Exported symbols stay global (`c`-prefix; no project namespace).
- Logging `"vaapivideo/<basename>: <msg>"`, never hot paths. Errors: FFmpeg `AvErr(ret).data()`, DRM/VAAPI `%m`/`strerror(errno)`.
- Members: trailing `///<`, brace-init; alphabetical within groups, but lifetime/lock order wins where it matters (e.g. a thread member declared last). Section rulers `// === LABEL ===`.
- `[[likely]]`/`[[unlikely]]` on hot/error paths; prefer early returns.
- Lambdas for tiny local glue; a named fn (anon-ns free or member) when reused, mutating, locking, calling a C API, or >~15 lines. Snapshot an atomic into a `const` before a lambda (avoid re-reads).

## Lifecycle (match existing patterns)
- `Clear()` resets buffers, keeps resources. `Shutdown()` releases everything + logs stats, idempotent via `stopping.exchange(true)`. `~Dtor() noexcept` calls `Shutdown()` if not already done.
