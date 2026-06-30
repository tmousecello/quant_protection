"""Shared per-flip JSONL writer for the Stage-1 runners (E1 / E3a / E3b).

The Stage-1 runbook promises a per-flip raw audit trail + rich log so a workstation run that fails
(no Claude on that host) can be debugged offline against the stub. All three runners stream every
measured flip through this single writer. The MEASUREMENT fields are identical across runners (they
all come from phase3_e1_vuln.apply_and_measure); each runner adds its own context (e1: byte/bit/
tag; e3a: phase/k/trial; e3b: mode/trial), so the rows share a schema core and differ only in the
context envelope. Centralizing the open/flush/close + done-marker here keeps that one writer (no
three inline copies to drift).
"""
import json
import os


class RawWriter:
    """Append-only JSONL writer with periodic flush and an optional completion marker.

    Usage:
        with RawWriter(path, done_path=dp) as w:
            for ...:
                w.write(rec)          # rec: a json-serializable dict
        # on clean exit the done marker is written (enables E1 --resume)
    """

    def __init__(self, path, done_path=None, flush_every=200):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self.done_path = done_path
        self._flush_every = flush_every
        self._fh = open(path, "w")
        self.n = 0

    def write(self, rec):
        self._fh.write(json.dumps(rec) + "\n")
        self.n += 1
        if self.n % self._flush_every == 0:
            self._fh.flush()
        return rec

    def close(self, mark_done=True):
        if self._fh.closed:
            return
        self._fh.flush()
        self._fh.close()
        if mark_done and self.done_path is not None:
            open(self.done_path, "w").close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        # mark complete only on a clean exit (so E1 --resume never reloads a half-written shard)
        self.close(mark_done=exc_type is None)
        return False
