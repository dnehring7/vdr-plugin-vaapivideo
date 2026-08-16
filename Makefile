# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dirk Nehring <dnehring@gmx.net>
#
# VAAPI Video plugin for the Video Disk Recorder
#

# The official name of this plugin.
# This name will be used in the '-P...' option of VDR to load the plugin.
# By default the main source file also carries this name.

PLUGIN = vaapivideo

### The version number of this plugin (taken from the main source file):

VERSION = $(shell sed -n 's/.*PLUGIN_VERSION[^"]*"\([^"]*\)".*/\1/p' src/common.h)

### The directory environment:

# Use package data if installed...otherwise assume the build runs under the VDR source directory:
PKG_CONFIG ?= pkg-config
PKGCFG = $(if $(VDRDIR),$(shell $(PKG_CONFIG) --variable=$(1) $(VDRDIR)/vdr.pc),$(shell PKG_CONFIG_PATH="$$PKG_CONFIG_PATH:../../.." $(PKG_CONFIG) --variable=$(1) vdr))
LIBDIR = $(call PKGCFG,libdir)
LOCDIR = $(call PKGCFG,locdir)
PLGCFG = $(call PKGCFG,plgcfg)
#
TMPDIR ?= /tmp

### The compiler options:

export CFLAGS   = $(call PKGCFG,cflags)
export CXXFLAGS = $(call PKGCFG,cxxflags)

### The version number of VDR's plugin API:

APIVERSION = $(call PKGCFG,apiversion)

### Allow user defined options to overwrite defaults:

-include $(PLGCFG)

# ------------------------------------------------------------
# Dependencies (pkg-config)
# ------------------------------------------------------------
REQUIRED_LIBS = alsa libavcodec libavfilter libavformat libavutil libdrm libswresample libva-drm

# Library Validation Function
define check_lib
$(shell $(PKG_CONFIG) --exists $(1) || (echo "Error: $(1) not found" >&2 && exit 1))
endef

# Validate Required Libraries
$(foreach lib,$(REQUIRED_LIBS),$(call check_lib,$(lib)))

# ------------------------------------------------------------
# Toolchain
# ------------------------------------------------------------
CXX ?= g++
CXXFLAGS ?= -s -O3 -march=native -mtune=native -flto=auto

# ------------------------------------------------------------
# Sanitizers -- `make SANITIZE=thread` (or =address); empty = packaging build.
# ------------------------------------------------------------
# TSan and ASan cannot coexist in one process. The flags REPLACE VDR's cxxflags:
# core's -O3/-flto would defeat -Og/-fno-lto (the dropped VDR defines are no-ops
# on LP64, so no ABI skew).
SANITIZE ?=

ifeq ($(SANITIZE),thread)
CXXFLAGS = -g -Og -fno-omit-frame-pointer -fno-lto \
           -fsanitize=thread \
           -ftrivial-auto-var-init=zero
else ifeq ($(SANITIZE),address)
CXXFLAGS = -g -Og -fno-omit-frame-pointer -fno-lto \
           -fsanitize=address,undefined,leak \
           -fsanitize-address-use-after-scope \
           -fstack-protector-strong \
           -ftrivial-auto-var-init=zero
else ifneq ($(SANITIZE),)
$(error SANITIZE must be one of: thread, address, or empty)
endif

# TSan runtime setup:
#   TSAN_OPTIONS="log_path=/var/tmp/vdr-tsan:log_exe_name=1:strip_path_prefix=/srv/vdr-packages:history_size=7:second_deadlock_stack=1:print_full_thread_history=1:malloc_context_size=10:memory_limit_mb=16384:exitcode=0:suppressions=/etc/vdr/tsan-suppressions.txt:print_suppressions=1"
#   LD_PRELOAD="/usr/lib64/libtsan.so.2"  # only when the core is NOT TSan-built (stock vdr)
#
# - TSAN_OPTIONS + suppressions live in /etc/sysconfig/vdr; reports land in
#   /var/tmp/vdr-tsan.vdr.<pid>. log_path is mandatory as daemon (stderr goes to
#   /dev/null); exitcode=0 keeps systemd from restart-looping on a finding.
# - Build VDR core with the same flags via Make.config in the core tree (never on
#   make's command line -- that silences the core Makefile's own `CXXFLAGS +=`).
#   The vdr binary then links libtsan itself. Without that, LD_PRELOAD the
#   runtime: pulled in first by dlopen it cannot allocate its static TLS block.
# - The suppressions file must exist, or libtsan aborts at startup.
#
# ASan run: LD_PRELOAD libasan/libubsan and e.g.
#   ASAN_OPTIONS="detect_leaks=1:abort_on_error=1:fast_unwind_on_malloc=0:detect_stack_use_after_return=1"
#   UBSAN_OPTIONS="print_stacktrace=1:halt_on_error=0"

# Development warnings: lenient by default (safe for packaging); enable with `make DEV_WARNINGS=1`.
DEV_WARNINGS ?= 0
ifeq ($(DEV_WARNINGS),1)
CXXFLAGS += -pedantic-errors -Wall -Wextra
CXXFLAGS += -Wformat=2 -Wconversion -Wsign-conversion -Wshadow -Werror -Wnull-dereference
endif

# libstdc++ debug mode -- bounds checking on iterators, vectors, strings
# WARNING: changes ABI; VDR and all plugins must be recompiled with this flag
#CXXFLAGS += -D_GLIBCXX_DEBUG -D_GLIBCXX_DEBUG_PEDANTIC

# GCC static analyzer (slow -- finds null-deref, use-after-free, double-free at compile time)
#CXXFLAGS += -fanalyzer

CXXFLAGS += -std=c++20 -fPIC
CXXFLAGS += -DPLUGIN_NAME_I18N='"$(PLUGIN)"'
CXXFLAGS += -I$(VDRDIR)/include -I.
CXXFLAGS += $(shell $(PKG_CONFIG) --cflags $(REQUIRED_LIBS))

ifeq ($(origin RPM_ARCH),undefined)
LDFLAGS := $(filter-out %redhat-package-notes,$(LDFLAGS))
endif

# Derived from the final CXXFLAGS, not from SANITIZE, so compile and link can
# never disagree (a sanitizer-built core also injects -fsanitize via vdr.pc's
# cxxflags). Adds only the DT_NEEDED -- the runtime must already be in the
# process (see the TSan setup notes above).
LDFLAGS += $(filter -fsanitize=%,$(CXXFLAGS))

SO_LDFLAGS = $(LDFLAGS) -shared -Wl,--no-as-needed

LDLIBS = $(shell $(PKG_CONFIG) --libs $(REQUIRED_LIBS)) -pthread

# ------------------------------------------------------------
# Sources / targets
# ------------------------------------------------------------
SOURCES = $(PLUGIN).cpp \
          src/audio.cpp \
          src/caps.cpp \
          src/config.cpp \
          src/decoder.cpp \
          src/device.cpp \
          src/display.cpp \
          src/filter.cpp \
          src/mediaplayer.cpp \
          src/osd.cpp \
          src/pes.cpp \
          src/stream.cpp \
          src/subtitle.cpp

# Build Artifacts
OBJECTS = $(SOURCES:.cpp=.o)
SOFILE = libvdr-$(PLUGIN).so
HEADERS = $(wildcard src/*.h)

# Build Targets
.PHONY: all clean install dist indent lint docs check-docs-defaults probe

all: $(SOFILE)

$(SOFILE): $(OBJECTS)
	$(CXX) $(SO_LDFLAGS) $(OBJECTS) -o $@ $(LDLIBS)

# Makefile in the prerequisites: a build-flag change (above all SANITIZE=) must invalidate
# every object. Without it, flipping the sanitizer relinks stale uninstrumented objects into
# the .so -- TSan then sees only part of the plugin, silently missing races in the untouched
# translation units while reporting bogus ones where a happens-before edge went unobserved.
%.o: %.cpp $(HEADERS) Makefile
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Standalone VAAPI capability prober
PROBE_BIN = vaapivideo-probe
PROBE_SRC = vaapivideo-probe.cpp
PROBE_PKGS = libdrm libva-drm

probe: $(PROBE_BIN)

$(PROBE_BIN): $(PROBE_SRC) Makefile
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(shell $(PKG_CONFIG) --cflags $(PROBE_PKGS)) \
		$< $(shell $(PKG_CONFIG) --libs $(PROBE_PKGS)) -o $@

.deps: $(SOURCES) $(HEADERS) Makefile
	$(CXX) -MM $(CXXFLAGS) $(SOURCES) > $@

# Include Dependencies (only for build targets, skip for clean/dist/docs/etc)
ifeq ($(filter clean dist docs check-docs-defaults lint indent,$(MAKECMDGOALS)),)
-include .deps
endif

# Format source code using clang-format
indent:
	@echo "Formatting source code..."
	@if command -v clang-format >/dev/null 2>&1; then \
        clang-format -i $(SOURCES) $(HEADERS); \
        echo "Code formatted with clang-format"; \
    else \
        echo "clang-format not found, skipping formatting"; \
    fi

clean:
	@-rm -f $(OBJECTS) $(SOFILE) $(PROBE_BIN) .deps compile_commands.json
	@-rm -f *.so *.tgz core* *~ src/*~
	@-rm -rf docs .lint

install: $(SOFILE)
	install -D $< $(DESTDIR)$(LIBDIR)/$<.$(APIVERSION)

dist: clean
	@-rm -f ../vdr-$(PLUGIN)-$(VERSION).tar.gz
	@cd .. && tar czf vdr-$(PLUGIN)-$(VERSION).tar.gz \
		--transform='s/^vdr-plugin-$(PLUGIN)/vdr-plugin-$(PLUGIN)-$(VERSION)/' \
		--exclude={.git,'*.tar.gz','*.o','*.so',.deps} \
		vdr-plugin-$(PLUGIN)
	@echo "Distribution package created as ../vdr-$(PLUGIN)-$(VERSION).tar.gz"

# A brace-init that clang-format wrapped (declaration + trailing ///< over ColumnLimit) makes
# Doxygen drop the default value from the rendered declaration -- silently, no warning. Catch the
# pattern "IDENT{" at EOL followed by a lone "value};": aggregates keep their comma or open "{{",
# so only the wrapped-scalar case matches. Fix by moving the comment above the member as "///".
check-docs-defaults:
	@awk '/[[:alnum:]_]\{[[:space:]]*$$/ { decl = $$0; file = FILENAME; line = FNR; next } \
	      decl != "" { if ($$0 ~ /^[[:space:]]*[^,{}]+\};/) { \
	          printf "%s:%d: wrapped default value -- Doxygen drops it:\n  %s\n  %s\n", \
	                 file, line, decl, $$0; rc = 1 } decl = "" } \
	      END { exit rc }' $(SOURCES) $(HEADERS) $(PROBE_SRC) \
	  || { echo "make docs: put the initializer on one line (see comment in Makefile)"; exit 1; }

docs: check-docs-defaults
	@command -v doxygen >/dev/null 2>&1 || { echo "doxygen not found"; exit 1; }
	@doxygen && echo "Doxygen documentation written to docs/html/index.html"

# Static Code Analysis using clang-tidy (checks configured in .clang-tidy)
# Filter out GCC-specific flags that clang doesn't understand.
CLANG_CXXFLAGS = $(filter-out -Wno-complain-wrong-lang -specs=% -grecord-gcc-switches -fanalyzer,$(CXXFLAGS))
PROBE_CLANG_CXXFLAGS = $(filter-out -I/usr/include/ffmpeg,$(CLANG_CXXFLAGS)) \
	$(shell $(PKG_CONFIG) --cflags $(PROBE_PKGS))

# Default parallel job count: nproc if available, else 1. Override with `make lint NJOBS=N`.
NJOBS ?= $(shell nproc 2>/dev/null || echo 1)

# Per-source stamp files. Touching the source re-lints just that file; running with -j$(NJOBS)
# spreads the per-file clang-tidy invocations across cores (clang-tidy itself is single-threaded).
LINT_STAMPS = $(SOURCES:%.cpp=.lint/%.stamp) $(PROBE_SRC:%.cpp=.lint/%.stamp)

# compile_commands.json is regenerated only when sources or headers change. The bear-driven full
# rebuild is the expensive step; without this dependency lint would pay ~15s on every invocation.
# Makefile is a dep so build-flag edits also invalidate the database.
compile_commands.json: $(SOURCES) $(HEADERS) $(PROBE_SRC) Makefile
	@command -v bear >/dev/null 2>&1 || { echo "bear not found"; exit 1; }
	@bear --force-preload -- $(MAKE) --no-print-directory -B -j$(NJOBS) $(OBJECTS)

# Check clang-tidy availability BEFORE running the expensive bear step, so a missing
# tool short-circuits without paying for the rebuild. compile_commands.json is invoked
# only after the check passes.
lint:
	@if ! command -v clang-tidy >/dev/null 2>&1; then \
		echo "clang-tidy not found"; \
		exit 0; \
	fi; \
	echo "Running clang-tidy analysis (NJOBS=$(NJOBS))..."; \
	$(MAKE) --no-print-directory compile_commands.json && \
	$(MAKE) --no-print-directory -j$(NJOBS) $(LINT_STAMPS)

# Probe uses libdrm/libva flags (specific rule). Plugin sources use the FFmpeg/VDR set
# (pattern rule). Make prefers the explicit rule for the probe stamp.
# .clang-tidy / Makefile in deps: a check-config or build-flag change invalidates the stamp
# so the next lint actually picks them up.
.lint/$(PROBE_SRC:%.cpp=%.stamp): $(PROBE_SRC) .clang-tidy Makefile
	@mkdir -p $(dir $@)
	@clang-tidy $< --quiet -- $(PROBE_CLANG_CXXFLAGS)
	@touch $@

# $(HEADERS) in deps: re-lint when any header changes (avoids stale results when a
# transitively included symbol moves or a definition changes). compile_commands.json in
# deps so a stamp invoked directly still rebuilds the database first.
.lint/%.stamp: %.cpp $(HEADERS) .clang-tidy compile_commands.json Makefile
	@mkdir -p $(dir $@)
	@clang-tidy $< --quiet -- $(CLANG_CXXFLAGS)
	@touch $@
