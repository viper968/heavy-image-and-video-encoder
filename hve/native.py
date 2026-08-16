"""Native C backend: the fastest of the three implementations of the codec.

The three paths, in the order the codec prefers them:

    hve/native.py + csrc/   C, threaded motion search   this file
    hve/fast.py             numba                       ~2x slower to encode
    hve/model.py            pure Python                 the reference

All three must produce byte-identical streams. `model.py` remains the
definition of the format; this file is an optimisation and nothing more, and
tests/test_native.py fails the build rather than shipping a divergence.

The library is compiled on first import with the system C compiler and cached
next to the package. There is no build step to run and no compiler needed at
install time — if `cc` is missing or the build fails, `available()` returns
False and the codec silently uses the numba path instead, which is why nothing
here raises on failure.

Two things this path does that numba cannot:

  - It threads the motion search. Every block's search is independent, so this
    is a pure encoder-side win with no format implications. The pixel loop is
    still serial and still the floor on encode time; threading *that* needs the
    slice-independent format change described in docs/research.md.
  - It narrows the scratch buffers. `fast.py` shares its arrays with the
    reference implementation's Python lists and so carries everything as int64;
    here the match model's two random-access tables are int32 and uint8, which
    at 1080p is 6 MB of working set instead of 24 MB.
"""

import ctypes
import os
import subprocess
import sysconfig
import tempfile

import numpy as np

from . import mix, model

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CSRC = os.path.join(_ROOT, "csrc")
_SOURCES = ("kernel.c", "motion.c")
_HEADERS = ("hve.h",)

# -fwrapv matters for correctness, not speed: signed overflow is undefined in C
# but wraps in numba's int64, and the mixer weights are the kind of unbounded
# accumulator where the two could differ. The kernel contains no floating point
# at all, so -march=native changes instruction selection but cannot change a
# single output byte — verified by hashing the stream both ways.
_BASE_CFLAGS = ["-fwrapv", "-fPIC", "-shared", "-pthread",
                "-fno-strict-aliasing", "-std=c11"]
# Tried in order; the first that compiles wins, so an older compiler or a
# cross-build that rejects -march=native still gets a working library.
_CFLAG_SETS = [["-O3", "-march=native"], ["-O3"], ["-O2"]]

_lib = None
_load_error = None


def _libname():
    return "_hve" + (sysconfig.get_config_var("EXT_SUFFIX") or ".so")


def _candidate_paths():
    """Next to the package, then a per-user temp dir for read-only installs."""
    yield os.path.join(_ROOT, "hve", _libname())
    uid = os.getuid() if hasattr(os, "getuid") else 0
    yield os.path.join(tempfile.gettempdir(), "hve-native-%d" % uid, _libname())


def _newest_source():
    return max(os.path.getmtime(os.path.join(_CSRC, f))
               for f in _SOURCES + _HEADERS)


def _compile(dest):
    cc = os.environ.get("CC") or "cc"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".%d.tmp" % os.getpid()
    last = "no compiler flag set was accepted"
    for opt in _CFLAG_SETS:
        cmd = ([cc] + opt + _BASE_CFLAGS
               + [os.path.join(_CSRC, f) for f in _SOURCES] + ["-o", tmp])
        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode == 0:
            os.replace(tmp, dest)
            return dest
        last = proc.stderr.decode()[-2000:]
    raise RuntimeError(last)


def _load():
    global _lib, _load_error
    if _lib is not None or _load_error is not None:
        return _lib
    if os.environ.get("HVE_NO_NATIVE"):
        _load_error = "disabled by HVE_NO_NATIVE"
        return None
    try:
        newest = _newest_source()
    except OSError as exc:                       # installed without csrc/
        newest = None
        _load_error = str(exc)
    for path in _candidate_paths():
        try:
            if (newest is not None
                    and (not os.path.exists(path)
                         or os.path.getmtime(path) < newest)):
                _compile(path)
            _lib = ctypes.CDLL(path)
            _bind(_lib)
            _load_error = None
            return _lib
        except Exception as exc:                 # try the next location
            _load_error = "%s: %s" % (path, exc)
    return None


def available():
    return _load() is not None


def load_error():
    """Why the native path is unavailable, for tools that want to report it."""
    _load()
    return _load_error


# --------------------------------------------------------------------------
# structures, mirroring csrc/hve.h

_I64 = ctypes.POINTER(ctypes.c_int64)
_I32 = ctypes.POINTER(ctypes.c_int32)
_U8 = ctypes.POINTER(ctypes.c_uint8)


class _Ladder(ctypes.Structure):
    _fields_ = [("v", _I64), ("n", ctypes.c_int64)]


class _RC(ctypes.Structure):
    _fields_ = [("s", ctypes.c_int64 * 5)]


class _Model(ctypes.Structure):
    # Field order must match csrc/hve.h exactly.
    _fields_ = (
        [(n, _I64) for n in ("zero_p", "dir_p", "diff_p", "match_p", "sign_p",
                             "nb_p", "nbm_p", "mant_p", "conf_p", "nbc_p",
                             "mixw", "nbmixw")]
        + [("lmsw", _I32)]
        + [(n, _I64) for n in ("apm0", "apm1", "apm2", "stretch", "squash")]
        + [(n, _Ladder) for n in ("act_l", "err_l", "lum_l", "side_l",
                                  "diff_l", "mexp_l", "conf_l", "adj_l")]
        + [("match_table", _I32), ("flat", _U8), ("errmap", _U8),
           ("errmap_stride", ctypes.c_int64), ("stats", _I64),
           ("params", _I64)])


class _Inter(ctypes.Structure):
    _fields_ = [("on", ctypes.c_int64), ("modes", _I64), ("mvs", _I64),
                ("nby", ctypes.c_int64), ("nbx", ctypes.c_int64),
                ("bs_y", ctypes.c_int64), ("bs_x", ctypes.c_int64),
                ("mv_sy", ctypes.c_int64), ("mv_sx", ctypes.c_int64),
                ("ref", _U8)]


def _bind(lib):
    lib.hve_code_plane.restype = ctypes.c_int
    lib.hve_code_plane.argtypes = [
        ctypes.c_int, _U8, ctypes.c_int64, ctypes.c_int64, _U8, _U8,
        ctypes.POINTER(_RC), ctypes.POINTER(_Model), ctypes.c_int64,
        ctypes.c_int64, ctypes.c_int64, ctypes.POINTER(_Inter)]
    lib.hve_code_block_info.restype = None
    lib.hve_code_block_info.argtypes = [
        ctypes.c_int, ctypes.POINTER(_RC), _U8, _U8, _I64, _I64, _I64, _I64,
        _I64, _I64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
    lib.hve_finish_encode.restype = ctypes.c_int64
    lib.hve_finish_encode.argtypes = [ctypes.POINTER(_RC), _U8]
    lib.hve_motion_search.restype = ctypes.c_int
    lib.hve_motion_search.argtypes = [
        _U8, _U8, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_int64, _I32, _I32, _I64, ctypes.c_int]
    lib.hve_spatial_cost.restype = ctypes.c_int
    lib.hve_spatial_cost.argtypes = [
        _U8, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, _I32, _I64,
        ctypes.c_int]
    lib.hve_halfpel_planes.restype = None
    lib.hve_halfpel_planes.argtypes = [_U8, ctypes.c_int64, ctypes.c_int64, _U8]
    lib.hve_threads_default.restype = ctypes.c_int
    lib.hve_threads_default.argtypes = []


def _p(arr, ptype):
    return arr.ctypes.data_as(ptype)


def threads():
    n = os.environ.get("HVE_THREADS")
    if n:
        return max(1, int(n))
    return _load().hve_threads_default()


# --------------------------------------------------------------------------
# the model bank


_BANK_KEYS = ("zero", "zero_dir", "zero_diff", "zero_match", "sign", "nb",
              "nb_match", "mant", "zero_conf", "nb_conf")
_FIELD_FOR_KEY = ("zero_p", "dir_p", "diff_p", "match_p", "sign_p", "nb_p",
                  "nbm_p", "mant_p", "conf_p", "nbc_p")
_LADDERS = (("act_l", "ACT_LADDER"), ("err_l", "ERR_LADDER"),
            ("lum_l", "LUMA_LADDER"), ("side_l", "SIDE_LADDER"),
            ("diff_l", "DIFF_LADDER"), ("mexp_l", "MEXP_LADDER"),
            ("conf_l", "CONF_LADDER"), ("adj_l", "ADJ_LADDER"))


class Bank:
    """The adaptive state plus the scratch buffers, as arrays C can mutate.

    Sizes and initial values come from `model.new_model()` — the same call the
    reference and numba paths use — so there is one definition of the model's
    shape and this file cannot drift from it by forgetting to resize a table.
    """

    def __init__(self, luma_h, luma_w, video=False):
        lib = _load()
        bank = model.new_model()
        self._keep = []
        self.m = _Model()

        for key, field in zip(_BANK_KEYS, _FIELD_FOR_KEY):
            self._set(field, np.array(bank[key], dtype=np.int64), _I64)
        self._set("mixw", np.array(bank["zero_mix"].weights, dtype=np.int64), _I64)
        self._set("nbmixw", np.array(bank["nb_mix"].weights, dtype=np.int64), _I64)
        self._set("apm0", np.array(bank["zero_apm"].table, dtype=np.int64), _I64)
        self._set("apm1", np.array(bank["nb_apm"].table, dtype=np.int64), _I64)
        self._set("apm2", np.array(bank["zero_apm2"].table, dtype=np.int64), _I64)
        self._set("lmsw", np.array(bank["lms"], dtype=np.int32), _I32)
        self._set("stretch", np.array(mix.STRETCH, dtype=np.int64), _I64)
        self._set("squash", np.array(mix.SQUASH, dtype=np.int64), _I64)

        for field, name in _LADDERS:
            arr = np.array(getattr(model, name), dtype=np.int64)
            self._keep.append(arr)
            setattr(self.m, field, _Ladder(_p(arr, _I64), len(arr)))

        self._set("match_table",
                  np.zeros(model.MATCH_HASH_MASK + 1, dtype=np.int32), _I32)
        self._set("flat", np.zeros(luma_h * luma_w, dtype=np.uint8), _U8)
        self._set("errmap", np.zeros((luma_h, luma_w), dtype=np.uint8), _U8)
        self.m.errmap_stride = luma_w
        self.stats = np.zeros(8, dtype=np.int64)
        self._set("stats", self.stats, _I64)
        from . import fast
        self.params = fast._params()
        self._set("params", self.params, _I64)

        if video:
            from . import video as _video
            vb = _video.new_video_model()
            self.mode_p = np.array(vb["mode"], dtype=np.int64)
            self.mv_zero = np.array(vb["mv_zero"], dtype=np.int64)
            self.mv_sign = np.array(vb["mv_sign"], dtype=np.int64)
            self.mv_mag = np.array(vb["mv_mag"], dtype=np.int64)
        self._lib = lib

    def _set(self, field, arr, ptype):
        self._keep.append(arr)
        setattr(self.m, field, _p(arr, ptype))

    def code(self, coder, encode, plane, kind, use_luma, write_errmap,
             inter=None):
        """Code one uint8 plane in place. `plane` must be C-contiguous."""
        if inter is None:
            iptr = None
        else:
            on, modes, mvs, bs_y, bs_x, mv_sy, mv_sx, ref = inter
            it = _Inter(on, _p(modes, _I64), _p(mvs, _I64),
                        modes.shape[0], modes.shape[1], bs_y, bs_x,
                        mv_sy, mv_sx, _p(ref, _U8))
            iptr = ctypes.byref(it)
        rcode = self._lib.hve_code_plane(
            1 if encode else 0, _p(plane, _U8), plane.shape[0], plane.shape[1],
            _p(coder.data, _U8), _p(coder.out, _U8), ctypes.byref(coder.rc),
            ctypes.byref(self.m), kind, use_luma, write_errmap, iptr)
        if rcode != 0:
            raise MemoryError("native kernel could not allocate row buffers")

    def code_block_info(self, coder, encode, modes, mvs, mv_max):
        self._lib.hve_code_block_info(
            1 if encode else 0, ctypes.byref(coder.rc), _p(coder.data, _U8),
            _p(coder.out, _U8), _p(self.mode_p, _I64), _p(self.mv_zero, _I64),
            _p(self.mv_sign, _I64), _p(self.mv_mag, _I64), _p(modes, _I64),
            _p(mvs, _I64), modes.shape[0], modes.shape[1], mv_max)


class Coder:
    """Range-coder state shared across planes and frames of one stream."""

    def __init__(self, encode, capacity=0, payload=None):
        self._lib = _load()
        self.encode = encode
        self.rc = _RC()
        self.rc.s[1] = 0xFFFFFFFF
        if encode:
            self.rc.s[3] = 1
            self.out = np.zeros(capacity, dtype=np.uint8)
            self.data = np.zeros(1, dtype=np.uint8)
        else:
            self.data = np.frombuffer(payload, dtype=np.uint8).copy()
            self.out = np.zeros(1, dtype=np.uint8)
            self.rc.s[0] = int.from_bytes(bytes(self.data[1:5]), "big")
            self.rc.s[2] = 5

    def finish(self):
        n = self._lib.hve_finish_encode(ctypes.byref(self.rc), _p(self.out, _U8))
        return bytes(self.out[:int(n)])


# --------------------------------------------------------------------------
# stills


def encode_planes(planes_u8):
    planes = np.ascontiguousarray(planes_u8, dtype=np.uint8)
    channels, height, width = planes.shape
    bank = Bank(height, width)
    coder = Coder(True, capacity=planes.size * 2 + 65536)
    for i in range(channels):
        bank.code(coder, True, planes[i], min(i, 3), 1 if i in (1, 2) else 0,
                  1 if i == 0 else 0)
    return coder.finish()


def decode_planes(payload, channels, height, width):
    planes = np.zeros((channels, height, width), dtype=np.uint8)
    bank = Bank(height, width)
    coder = Coder(False, payload=payload)
    for i in range(channels):
        bank.code(coder, False, planes[i], min(i, 3), 1 if i in (1, 2) else 0,
                  1 if i == 0 else 0)
    return planes


# --------------------------------------------------------------------------
# motion estimation


def halfpel_planes(ref):
    ref = np.ascontiguousarray(ref, dtype=np.uint8)
    h, w = ref.shape
    out = np.empty((4, h, w), dtype=np.uint8)
    _load().hve_halfpel_planes(_p(ref, _U8), h, w, _p(out, _U8))
    return out


def motion_search(cur, ref, bs, search, cost_table, halfpel_bias,
                  pyramid_min_pixels, pyramid_levels, refine_radius):
    cur = np.ascontiguousarray(cur, dtype=np.uint8)
    ref = np.ascontiguousarray(ref, dtype=np.uint8)
    h, w = cur.shape
    nby, nbx = -(-h // bs), -(-w // bs)
    mv = np.zeros((nby, nbx, 2), dtype=np.int32)
    cost = np.zeros((nby, nbx), dtype=np.int64)
    tbl = np.ascontiguousarray(cost_table, dtype=np.int32)
    rcode = _load().hve_motion_search(
        _p(cur, _U8), _p(ref, _U8), h, w, bs, search, halfpel_bias,
        pyramid_min_pixels, pyramid_levels, refine_radius, _p(tbl, _I32),
        _p(mv, _I32), _p(cost, _I64), threads())
    if rcode != 0:
        raise MemoryError("native motion search could not allocate a pyramid")
    return mv, cost


def spatial_cost(cur, bs, cost_table):
    cur = np.ascontiguousarray(cur, dtype=np.uint8)
    h, w = cur.shape
    nby, nbx = -(-h // bs), -(-w // bs)
    cost = np.zeros((nby, nbx), dtype=np.int64)
    tbl = np.ascontiguousarray(cost_table, dtype=np.int32)
    _load().hve_spatial_cost(_p(cur, _U8), h, w, bs, _p(tbl, _I32),
                             _p(cost, _I64), threads())
    return cost
