"""The .hvi still-image codec: RCT -> MED prediction -> adaptive context coding."""

import numpy as np

from . import model, native, rc
from .bitio import Reader, Writer
from .transform import predict_plane, rct_forward, rct_inverse, zigzag

MAGIC = b"HVI4"
FLAG_RCT = 1

# Which model bank each plane uses: luma, Cb, Cr, alpha.
PLANE_KINDS = (0, 1, 2, 3)


def _residual_entropy(planes):
    """Zeroth-order entropy of the prediction residuals, in bytes. A cheap proxy."""
    total = 0.0
    for plane in planes:
        syms = zigzag((plane.astype(np.int32) - predict_plane(plane)) & 255)
        hist = np.bincount(syms.ravel(), minlength=256).astype(np.float64)
        used = hist > 0
        p = hist[used] / hist.sum()
        total += float(-(hist[used] * np.log2(p)).sum())
    return total / 8.0


def _planes_from_image(img):
    """Pick the colour transform that actually helps this image.

    The RCT is a large win on photographs, where the channels track each other,
    but it can be a large *loss* on synthetic images whose channels are unrelated
    or inversely related. One entropy estimate each is cheap, so measure instead
    of assuming.
    """
    if img.ndim == 2:
        return img[None, ...], 0
    c = img.shape[2]
    if c not in (3, 4):
        return img.transpose(2, 0, 1), 0

    plain = img[..., :3].transpose(2, 0, 1)
    transformed = rct_forward(img[..., :3])
    if _residual_entropy(transformed) <= _residual_entropy(plain):
        chosen, flags = transformed, FLAG_RCT
    else:
        chosen, flags = plain, 0
    if c == 4:
        chosen = np.concatenate([chosen, img[..., 3][None, ...]])
    return chosen, flags


def _image_from_planes(planes, channels, flags):
    if channels == 1:
        return planes[0]
    if flags & FLAG_RCT:
        rgb = rct_inverse(planes[:3])
        if channels == 4:
            return np.concatenate([rgb, planes[3][..., None]], axis=-1)
        return rgb
    return np.stack(planes, axis=-1)


def _check_features(features):
    """The pure-Python reference implements only the full model.

    A file coded with a reduced preset is still perfectly valid, but model.py
    does not know how to reproduce it, so refuse loudly rather than decode it
    wrongly and report success.
    """
    if features != model.FEAT_ALL and not native.available():
        raise RuntimeError(
            "this file uses model preset 0x%02x, and the pure-Python path only "
            "implements the full model (0x%02x). Build the C backend "
            "(make -C csrc) to read it." % (features, model.FEAT_ALL))


def _encode_payload(planes, width, height, features):
    """Range-coder payload for the planes, through the C kernel when it built.

    Both paths emit the same bytes — `tests/test_native.py` pins that exactly —
    so which one runs is purely a speed question and never a format question.
    """
    _check_features(features)
    if native.available():
        return native.encode_planes(planes, features=features)
    coder = rc.Encoder()
    bank = model.new_model()
    luma_err = None
    for i, plane in enumerate(planes):
        _, err = model.code_plane(coder, True, width, height, PLANE_KINDS[i], bank,
                                  src=plane.tolist(),
                                  luma_err=luma_err if i in (1, 2) else None)
        if i == 0:
            luma_err = err
    return coder.finish()


def _decode_payload(payload, channels, width, height, features):
    _check_features(features)
    if native.available():
        return list(native.decode_planes(payload, channels, height, width,
                                        features=features))
    coder = rc.Decoder(payload)
    bank = model.new_model()
    planes = []
    luma_err = None
    for i in range(channels):
        rows, err = model.code_plane(coder, False, width, height, PLANE_KINDS[i], bank,
                                     luma_err=luma_err if i in (1, 2) else None)
        planes.append(np.array(rows, dtype=np.uint8))
        if i == 0:
            luma_err = err
    return planes


def encode(img, features=None):
    """Compress an image array (HxW, HxWx3 or HxWx4, uint8) to .hvi bytes.

    `features` selects which model stages run; it is written into the header
    so the decoder needs no matching argument. See model.FEAT_*.
    """
    features = model.FEATURES if features is None else features
    img = np.ascontiguousarray(img, dtype=np.uint8)
    height, width = img.shape[:2]
    channels = 1 if img.ndim == 2 else img.shape[2]
    planes, flags = _planes_from_image(img)

    w = Writer()
    w.raw(MAGIC)
    w.varint(width)
    w.varint(height)
    w.u8(channels)
    w.u8(flags)
    w.u8(features)

    payload = _encode_payload(np.ascontiguousarray(planes), width, height,
                              features)
    w.varint(len(payload))
    w.raw(payload)
    return w.bytes()


def decode(data):
    """Decompress .hvi bytes back to the original array."""
    r = Reader(data)
    if r.raw(4) != MAGIC:
        raise ValueError("not an .hvi stream")
    width = r.varint()
    height = r.varint()
    channels = r.u8()
    flags = r.u8()
    features = r.u8()
    payload_len = r.varint()
    payload = r.raw(payload_len)

    planes = _decode_payload(payload, channels, width, height, features)
    return _image_from_planes(planes, channels, flags)
