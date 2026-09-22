#!/usr/bin/env bash
# Ten-second sanity run for notebook 025: one date, one instrument family,
# every pricing method, so a broken pricer surfaces before a full phase
# burns wall-clock. See NEXT_PROMPT.md section 6.
#
# Usage: scripts/smoke_025.sh
set -euo pipefail
cd "$(dirname "$0")/.."

.venv/bin/python -c "
import sys, time
sys.path.insert(0, 'src/research/tmp')
import numpy as np
import lib25
import pricers_25_barrier as pba
import pricers_25_asian as pas
import pricers_25_american as pam
import pricers_25_exotic as pex

t0 = time.perf_counter()
F, K, B, T, sigma, r = 70.0, 68.0, 55.0, 1.0, 0.30, 0.03
df = np.exp(-r * T)

checks = []
checks.append(('black76', lib25.black76(F, K, T, sigma, df, 'call')))
checks.append(('bachelier', lib25.bachelier(F, K, T, sigma * F, df, 'call')))
checks.append(('displaced_black', lib25.displaced_black(F, K, T, sigma, df, 'call', shift=5.0)))

paths = lib25.simulate_gbm(F, sigma, T, n_steps=20, n_paths=2000, seed=25)
checks.append(('barrier_analytic', pba.barrier_analytic(F, K, B, T, sigma, df, 'call', 'do')))
checks.append(('barrier_mc', pba.barrier_mc(paths, K, B, 'do', 'call', df)['price']))
checks.append(('barrier_pde', pba.barrier_pde(F, K, B, T, sigma, r, 'call', 'do', n_s=60, n_t=60)))

checks.append(('asian_geometric', pas.asian_geometric(F, K, T, sigma, df, 'call', 0.9, 5)))
checks.append(('asian_turnbull_wakeman', pas.asian_turnbull_wakeman(F, K, T, sigma, df, 'call', 0.9, 5)))

checks.append(('american_binomial', pam.american_binomial(F, K, T, sigma, r, 'put', n_steps=50)))
paths_small = lib25.simulate_gbm(F, sigma, T, n_steps=20, n_paths=2000, seed=25)
checks.append(('american_lsm', pam.american_lsm(paths_small, K, 'put', df)['price']))

checks.append(('margrabe', pex.margrabe(F, K, T, sigma, sigma * 1.1, 0.5, df)))
checks.append(('kirk', pex.kirk(F, K, 2.0, T, sigma, sigma * 1.1, 0.5, df, 'call')))
checks.append(('quanto_black76', pex.quanto_black76(F, K, T, sigma, 0.12, 0.4, df, 'call')))

elapsed = time.perf_counter() - t0
ok = True
for name, val in checks:
    good = np.isfinite(val) and val >= -1e-6
    ok = ok and good
    print(f\"{'OK' if good else 'FAIL'}  {name:28s} = {val:.4f}\")
print(f'\\n{elapsed:.2f}s elapsed, all_ok={ok}')
sys.exit(0 if ok else 1)
"
