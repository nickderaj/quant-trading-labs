"""Notebook 017 Phase 3 (NEXT_PROMPT.md sec 9.2): emits one line per grid
cell as `N T rho skew kurt mode`, read from Phase 0's pre-registration JSON
so the grid and the pre-registration cannot drift apart. Both null (DS-2)
and edge (DS-3) modes for every (N, T, rho, moments) combination -- sec
4.1: "each run for all variants under the null (DS-2) and again with an
injected edge (DS-3)".
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
