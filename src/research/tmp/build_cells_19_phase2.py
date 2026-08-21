"""Notebook 019 Phase 2 (NEXT_PROMPT.md sec 4 row 2): emits one line per
design point as `N T rho skew kurt mode`, read from 017's OWN frozen grid
axes (src/research/tmp/phase_0_17_preregistration.json) so 019's profile
grid cannot drift from the grid Phase 3's prediction will need to match
cells against. Both modes (null, edge), per sec 4's phase-2 row.
"""

import json

with open("src/research/tmp/phase_0_17_preregistration.json") as f:
    prereg = json.load(f)

grid = prereg["grid"]
for n_trials in grid["N"]:
    for n_obs in grid["T"]:
        for rho in grid["rho"]:
            for m in grid["moments"]:
                for mode in ("null", "edge"):
                    print(
                        f"{n_trials} {n_obs} {rho} {m['skew']} {m['kurtosis']} {mode}"
                    )
