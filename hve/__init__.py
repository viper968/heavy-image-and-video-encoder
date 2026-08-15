"""hve - a heavy lossless encoder for images and video.

Compresses harder than PNG / lossless WebP / lossless AVIF by spending decode
time instead of bits. Output is not viewable in any standard viewer; you send it
along with this decoder and get the original bytes back.
"""

from . import image, video  # noqa: F401

__all__ = ["image", "video"]
__version__ = "0.1.0"
