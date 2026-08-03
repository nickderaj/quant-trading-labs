"""Phase 0: reproduction check (checkpoint 0).

Before extending notebook 5's tail-risk work, confirm the numbers being
extended are actually what the committed JSONs say. Re-derives three
published headline numbers from src/results/005_tail_risk_evt.md directly from
the committed notebook-5 result JSONs (not recomputed from scratch - this is
a check that the write-up matches the artifact, not a re-run of the
underlying models) and asserts each. Per NEXT_RUN_PROMPT.md section 1's own
stop condition: if any assertion fails, this script's non-zero exit / raised
AssertionError IS the stop-and-report signal - do not patch the number here.
"""

import json
import sys

sys.path.insert(0, "src/research/tmp")
sys.path.insert(0, "src")

PHASE3 = json.load(open("src/research/tmp/phase3_density_results.json"))
PHASE4 = json.load(open("src/research/tmp/phase4_coverage_results.json"))

# ---- 1. GARCH-t's 12h log score (2.623 in the write-up) ----
garch_t_12h = PHASE3["intervals"]["12h"]["scores"]["d5_garch_t"]["log_score_mean"]
print(f"GARCH-t 12h log score: {garch_t_12h:.3f} (write-up: 2.623)")
assert abs(garch_t_12h - 2.623) < 0.001, f"GARCH-t 12h log score {garch_t_12h} does not match write-up's 2.623"

# ---- 2. count of models clearing all 36 coverage tests at 12h (should be
# exactly 1, GARCH-EVT) ----
gate_b_12h = PHASE4["intervals"]["12h"]["gate_b_verdict"]
clearers_12h = [name for name, cleared in gate_b_12h.items() if cleared]
print(f"12h models clearing all 36 coverage tests: {clearers_12h}")
assert len(clearers_12h) == 1, f"expected exactly 1 clearer at 12h, got {len(clearers_12h)}: {clearers_12h}"
assert clearers_12h[0] == "d8_garch_evt", f"expected GARCH-EVT (d8) to be the sole 12h clearer, got {clearers_12h[0]}"

# ---- 3. the Gate A verdict at 1d (no significant winner) ----
gate_a_1d = PHASE3["intervals"]["1d"]["gate_a_verdict"]
print(f"1d Gate A verdict: best={gate_a_1d['best_by_log_score']} "
      f"significant(bootstrap)={gate_a_1d['beats_every_other_significantly_bootstrap_bh']}")
assert gate_a_1d["beats_every_other_significantly_bootstrap_bh"] is False, \
    "expected no significant Gate A winner at 1d, but the bootstrap-BH verdict says one fired"

print("\nAll three notebook-5 headline numbers reproduced from the committed JSONs. Proceeding.")
