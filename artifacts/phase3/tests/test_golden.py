"""Golden-file regression (Stage 0 component 4, RaBitQ-agnostic part).

Re-runs the code-generated fixture and asserts it reproduces the committed golden_smoke.json.
A diff here means a measurement component changed behaviour; if intended, regenerate with
`python artifacts/phase3/make_golden.py` and commit the new golden.
"""
import json
import os

import make_golden


def test_golden_reproduces():
    assert os.path.isfile(make_golden.GOLDEN), "golden missing; run make_golden.py"
    with open(make_golden.GOLDEN) as fh:
        committed = json.load(fh)
    fresh = make_golden.compute()
    # round-trip through json so tuples/np types compare equal to the committed (list) form
    fresh = json.loads(json.dumps(fresh))
    assert fresh == committed, "compute() drifted from the committed golden fixture"
