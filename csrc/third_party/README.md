# Vendored third-party code

## lodepng

PNG encoding and decoding, so the `hve` binary can read and write PNG without
linking libpng and zlib. That matters for shipping: a self-contained executable
for Linux and Windows should not need the target machine to have image libraries
installed, and cross-compiling libpng for Windows is more work than vendoring
one file.

- Upstream: https://github.com/lvandeve/lodepng
- Version: 20260119
- Licence: zlib (permissive; see the header of `lodepng.c`)
- Modifications: **none**. The file is `lodepng.cpp` renamed to `.c` — LodePNG
  is written in C, with its C++ wrappers behind `LODEPNG_COMPILE_CPP`, which the
  build does not define.

Nothing in `csrc/` outside this directory is third-party. If you update this,
re-run the test suite: `tests/test_cli_binary.py` checks that a PNG written by
the binary reloads to identical pixels.
