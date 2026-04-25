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

# Use package data if installed...otherwise assume we're under the VDR source directory:
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

# C++ Compiler Flags
CXXFLAGS ?= -s -O3 -march=native -mtune=native -flto=auto

# Debugging Flags for AddressSanitizer, LeakSanitizer, and UndefinedBehaviorSanitizer
# (uncomment for development builds — mutually exclusive with ThreadSanitizer)
#CXXFLAGS = -g -Og -fno-omit-frame-pointer -fno-lto \
#           -fsanitize=address,undefined,leak \
#           -fsanitize-address-use-after-scope \
#           -fstack-protector-strong \
#           -ftrivial-auto-var-init=zero
# Runtime options (copy into shell before starting VDR):
#   export ASAN_OPTIONS="detect_leaks=1:abort_on_error=1:symbolize=1:fast_unwind_on_malloc=0:strict_init_order=1:check_initialization_order=1:detect_stack_use_after_return=1"
#   export UBSAN_OPTIONS="print_stacktrace=1:halt_on_error=0"
#   export LD_PRELOAD="$(gcc -print-file-name=libasan.so):$(gcc -print-file-name=libubsan.so)"

# Debugging Flags for ThreadSanitizer
# (uncomment for development builds — mutually exclusive with AddressSanitizer)
#CXXFLAGS = -g -Og -fno-omit-frame-pointer -fno-lto \
#           -fsanitize=thread \
#           -ftrivial-auto-var-init=zero
# Runtime options (copy into shell before starting VDR):
#   export TSAN_OPTIONS="halt_on_error=0:second_deadlock_stack=1:detect_deadlocks=1:report_thread_leaks=1:history_size=7:symbolize=1"
#   export LD_PRELOAD="$(gcc -print-file-name=libtsan.so)"

# Development Flags (uncomment for development builds)
#CXXFLAGS += -pedantic-errors -Wall -Wextra
#CXXFLAGS += -Wformat=2 -Wconversion -Wsign-conversion -Wshadow -Werror -Wnull-dereference

# libstdc++ debug mode — bounds checking on iterators, vectors, strings
# WARNING: changes ABI; VDR and all plugins must be recompiled with this flag
#CXXFLAGS += -D_GLIBCXX_DEBUG -D_GLIBCXX_DEBUG_PEDANTIC

# GCC static analyzer (slow — finds null-deref, use-after-free, double-free at compile time)
#CXXFLAGS += -fanalyzer

CXXFLAGS += -std=c++20 -fPIC
CXXFLAGS += -DPLUGIN_NAME_I18N='"$(PLUGIN)"'
CXXFLAGS += -I$(VDRDIR)/include -I.
CXXFLAGS += $(shell $(PKG_CONFIG) --cflags $(REQUIRED_LIBS))

LDFLAGS := -shared -Wl,--no-as-needed

# Debugging Flags for AddressSanitizer, LeakSanitizer, and UndefinedBehaviorSanitizer
#LDFLAGS += -fsanitize=address,undefined,leak
# Debugging Flags for ThreadSanitizer
#LDFLAGS += -fsanitize=thread

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
          src/osd.cpp \
          src/pes.cpp \
          src/stream.cpp

# Build Artifacts
OBJECTS = $(SOURCES:.cpp=.o)
SOFILE = libvdr-$(PLUGIN).so
HEADERS = $(wildcard src/*.h)

# Build Targets
.PHONY: all clean install dist indent lint docs probe

all: $(SOFILE)

$(SOFILE): $(OBJECTS)
	$(CXX) $(LDFLAGS) $(OBJECTS) -o $@ $(LDLIBS)

%.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Standalone VAAPI capability prober. Links only libdrm + libva-drm — keep this
# rule separate from the plugin build so the probe binary does not accrue the
# plugin's full dependency closure (ffmpeg, alsa, ...).
PROBE_BIN = vaapivideo-probe
PROBE_SRC = vaapivideo-probe.cpp
PROBE_PKGS = libdrm libva-drm

probe: $(PROBE_BIN)

$(PROBE_BIN): $(PROBE_SRC)
	$(CXX) $(CXXFLAGS) $(shell $(PKG_CONFIG) --cflags $(PROBE_PKGS)) \
		$< $(shell $(PKG_CONFIG) --libs $(PROBE_PKGS)) -o $@

.deps: $(SOURCES) $(HEADERS)
	$(CXX) -MM $(CXXFLAGS) $(SOURCES) > $@

# Include Dependencies (only for build targets, skip for clean/dist/docs/etc)
ifeq ($(filter clean dist docs lint indent,$(MAKECMDGOALS)),)
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
	@-rm -rf docs

install: $(SOFILE)
	install -D $< $(DESTDIR)$(LIBDIR)/$<.$(APIVERSION)

dist: clean
	@-rm -f ../vdr-$(PLUGIN)-$(VERSION).tar.gz
	@cd .. && tar czf vdr-$(PLUGIN)-$(VERSION).tar.gz \
		--transform='s/^vdr-plugin-$(PLUGIN)/vdr-plugin-$(PLUGIN)-$(VERSION)/' \
		--exclude={.git,'*.tar.gz','*.o','*.so',.deps} \
		vdr-plugin-$(PLUGIN)
	@echo "Distribution package created as ../vdr-$(PLUGIN)-$(VERSION).tar.gz"

docs:
	@command -v doxygen >/dev/null 2>&1 || { echo "doxygen not found"; exit 1; }
	@doxygen && echo "Doxygen documentation written to docs/html/index.html"

# Static Code Analysis using clang-tidy (checks configured in .clang-tidy)
# Filter out GCC-specific flags that clang doesn't understand
CLANG_CXXFLAGS = $(filter-out -Wno-complain-wrong-lang -specs=% -grecord-gcc-switches,$(CXXFLAGS))

lint:
	@echo "Running clang-tidy analysis..."
	@command -v clang-tidy >/dev/null 2>&1 || { echo "clang-tidy not found"; exit 0; }
	@command -v bear >/dev/null 2>&1 || { echo "bear not found"; exit 1; }
	@bear --force-preload -- $(MAKE) --no-print-directory -B $(OBJECTS)
	@clang-tidy $(SOURCES) --quiet -- $(CLANG_CXXFLAGS)
