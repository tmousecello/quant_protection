"""Adapter factory — config-driven stub<->real switch for the RaBitQ runners.

The E1/E3a/E3b runners must run unchanged on two machines: the arm64 dev box (no RaBitQ binaries
-> the deterministic `stub_adapter`) and the x86 workstation (binaries built -> the live `adapter`).
`get_adapter()` resolves which module to use so the runner just does `adapter = get_adapter(...)`
and calls `adapter.search_corrupted(...)` etc. — no code change moving dev -> workstation.

Selection precedence:
  1. explicit `name` argument (CLI --adapter {stub,real,auto});
  2. else env var QP_RABITQ_ADAPTER (same os.environ override style as adapter.RABITQ_REPO);
  3. else auto: real if the x86 binaries are built, otherwise stub.

Guard: an explicit `real` request on a host WITHOUT binaries report-and-stops (reuses the real
adapter's message) — it NEVER silently downgrades to stub, because a "real" run that quietly
produced stub numbers is the worst possible failure for a research result.
"""
import os

from qp.rabitq import adapter as _real
from qp.rabitq import stub_adapter as _stub

VALID = ("stub", "real", "auto")


def get_adapter(name=None):
    """Return the adapter module (real or stub) per the precedence above."""
    choice = (name or os.environ.get("QP_RABITQ_ADAPTER") or "auto").lower()
    if choice not in VALID:
        raise ValueError(f"--adapter must be one of {VALID}, got {choice!r}")

    if choice == "stub":
        return _stub
    if choice == "real":
        _real._require_binaries()                     # report-and-stop if not built; never downgrade
        return _real
    # auto
    return _real if _real.binaries_built() else _stub


def adapter_name(mod):
    """'stub' or 'real' for the resolved module (for logging / record provenance)."""
    return "stub" if mod is _stub else "real"
