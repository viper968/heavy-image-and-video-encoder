# Build the standalone `hve` binary.
#
#   make                 tuned for this machine   -> build/hve
#   make portable        runs on any x86-64       -> build/hve
#   make static          no dynamic deps at all   -> build/hve   (for shipping)
#   make windows         cross-compile            -> build/hve.exe
#   make clean
#
# Nothing here needs Python, libpng or zlib: PNG comes from the vendored
# lodepng in third_party/, and csrc/model_constants.h is generated but checked
# in, so a fresh clone builds with a C compiler and nothing else.

CC      ?= cc
VERSION ?= 0.1.0
BUILD   ?= ../build

# -fwrapv is correctness, not speed: signed overflow is undefined in C but the
# reference implementation's integers are arbitrary precision.
CFLAGS_COMMON = -std=c11 -O3 -fwrapv -fno-strict-aliasing \
                -Wall -Wextra -Wno-unused-parameter \
                -DHVE_VERSION=\"$(VERSION)\"
# -march=native is safe only because the kernel contains no floating point at
# all, so instruction selection cannot change an output byte. A binary you
# intend to hand to someone else must not use it - build `portable` or `static`.
CFLAGS  ?= $(CFLAGS_COMMON) -march=native
LDFLAGS ?=
LDLIBS  ?= -lm -pthread

SRC = main.c container.c model.c transform.c util.c y4m.c imageio.c \
      kernel.c motion.c third_party/lodepng.c
HDR = hve.h hvefmt.h hvemodel.h med.h thread.h model_constants.h

.PHONY: all portable static windows clean

all: $(BUILD)/hve

$(BUILD)/hve: $(SRC) $(HDR)
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) $(SRC) -o $@ $(LDFLAGS) $(LDLIBS)

# Both of these change only the flags, so the existing binary is not out of
# date by make's reckoning and would be left alone. Build them directly.
portable:
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS_COMMON) $(SRC) -o $(BUILD)/hve $(LDFLAGS) $(LDLIBS)

static:
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS_COMMON) -static $(SRC) -o $(BUILD)/hve $(LDFLAGS) -lm

# A single .exe with no DLL dependencies beyond the system ones. The threading
# shim in thread.h uses the Win32 API directly, so winpthreads is not needed.
#
# __USE_MINGW_ANSI_STDIO makes printf understand %zu and %lld. Without it mingw
# hands those to the old msvcrt runtime, which does not know them, and the
# sizes in `hve info` come out as literal "%zu".
#
# Install the toolchain with one of:
#   Void      sudo xbps-install -S cross-x86_64-w64-mingw32 \
#                                  cross-x86_64-w64-mingw32-crt
#   Debian    sudo apt install gcc-mingw-w64-x86-64
#   Fedora    sudo dnf install mingw64-gcc
#
# NOTE: this target has never been run. The machine this was developed on has
# no mingw-w64 and no way to install one, so the Windows build is written to be
# correct and is unverified. Report anything it gets wrong.
WINCC ?= x86_64-w64-mingw32-gcc
windows: $(SRC) $(HDR)
	@mkdir -p $(BUILD)
	$(WINCC) $(CFLAGS_COMMON) -D__USE_MINGW_ANSI_STDIO=1 \
	    -static -static-libgcc $(SRC) -o $(BUILD)/hve.exe -lm

clean:
	rm -rf $(BUILD)
