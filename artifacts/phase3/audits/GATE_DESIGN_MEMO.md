# Gate-Design Memo (engineering; not for the paper)

Process record from the Stage 3 adversarial audits (2026-07-10). Kept alongside the two audit
rounds (`STAGE3_AUDIT.md` = round 1, `STAGE3_AUDIT_ROUND2.md` = round 2) so the reasoning survives.

## The rule

**A gate you cannot state a fail condition for is not a gate.** A "passing" check that is true *by
construction* — that no reachable input could have made fail — is evidence of nothing. It reads as
green and hides the absence of a real test.

## Why this memo exists

Round 1 returned "8/8 CLEAN". Round 2 overturned it. Two of the round-1 "passes" were structurally
unfailable, and one main-figure comparison was a two-variable confound that a mere file-existence
check waved through. Concretely, this project produced **three** by-construction "successes" that
looked like evidence but were not:

1. **run 3 "immortal" replicas** — the R=3 copies lived in independent memory and were never
   injected, so `cliff_irrecoverable=0` was guaranteed; it said nothing about majority-vote
   robustness. (Flagged in `E3C_E5_ANALYSIS.md` §4.4 at the time.)
2. **the Phase A `cliff_irrecoverable=0` gate** — under self-scrub semantics that counter is
   unreachable; the gate could not fail. Removed from the Stage 3 evidence chain (real evidence:
   `cliff_vote_fail_reloads=4`, recall flat at 0.98376, `cliff_repaired=2012`).
3. **the G "diagonal"** — `observed_load == injected f` to `max|dev|=0` is pinned by an in-run
   assertion (numerator = `crc_fail==n_elements`), not a measurement of detection accuracy. Reframed
   as a *by-construction* property: the detection layer emits the scheduling variable f for free on
   the service path.

## What we changed as a result

- Removed by-construction gates from the evidence chain; where the property is still worth stating,
  it is labelled "by construction / 構造驗證", never "accuracy" or "逐點相等 as a finding".
- Upgraded the main-figure input check in `phase3_f_synth.py` from *file-exists* to a real
  **lane-consistency gate**: the two replica-injecting lines (fuse, self_scrub) must carry the same
  `replica_lane` marker or the synth aborts (REPORT-AND-STOP). This gate *can* fail — point it at the
  old unmarked XOR-lane file and it does — which is the whole point. The marker is stamped by
  `phase3_e5_recovery.py` (`REPLICA_LANE`) into every `--inject-replicas` record row and summary.

## Checklist for future gates

- State the input that would make this gate FAIL. If you can't, it isn't a gate — delete it or make
  it one.
- Distinguish "measured and could have been otherwise" from "true by construction". Only the former
  is evidence.
- For any paired comparison, verify the two arms differ in exactly one variable (here: lane + seed
  was the hidden second variable). Encode that check in code, not in a reviewer's memory.
