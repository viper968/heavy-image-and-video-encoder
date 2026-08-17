"""csrc/model_constants.h is generated; a stale copy is a silent divergence.

The C program reads every tunable, ladder and lookup table from that header. If
someone changes a constant in `hve/model.py` and forgets to re-run the
generator, the binary keeps using the old model: it still produces valid,
losslessly-decodable files, just slightly worse ones, and nothing complains.
This is the thing that complains.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))


def test_generated_header_is_up_to_date():
    import gen_model_constants
    if not os.path.exists(gen_model_constants.OUT):
        pytest.fail("csrc/model_constants.h is missing; run "
                    "tools/gen_model_constants.py")
    with open(gen_model_constants.OUT) as fh:
        on_disk = fh.read()
    assert on_disk == gen_model_constants.generate(), (
        "csrc/model_constants.h is stale — run "
        ".venv/bin/python tools/gen_model_constants.py")
