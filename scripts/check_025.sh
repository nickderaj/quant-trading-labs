#!/usr/bin/env bash
# Referee for notebook 025. Prints the Phase 2 cross-validation matrix AND a
# set of invariants over the Phase 5/6 case studies, so the model never has to
# open the JSON to know whether the notebook is sound.
#
# Phase 2 alone used to be the whole referee. That is why a batch of case-study
# bugs -- a hedge with the wrong sign, a "seasonal" vol with no seasonality in
# it, a probability divided by its own observation count, a barrier that could
# never be touched -- all shipped while the banner still read 17/17 PASS. The
# checks below are the ones that would have caught them.
#
# Usage: scripts/check_025.sh
set -euo pipefail
cd "$(dirname "$0")/.."

P2="src/research/tmp/phase_2_25_crossval.json"
P5="src/research/tmp/phase_5_25_cases_abcd.json"
P6="src/research/tmp/phase_6_25_cases_efg.json"

for f in "$P2" "$P5" "$P6"; do
  if [ ! -f "$f" ]; then
    echo "no $f yet -- run scripts/run_025.sh first"
    exit 1
  fi
done

.venv/bin/python -c "
import json, math, sys

p2 = json.load(open('$P2'))
p5 = json.load(open('$P5'))
p6 = json.load(open('$P6'))

print('=== Phase 2: pricer cross-validation ===')
for c in p2['matrix']:
    print(f\"{'PASS' if c['passed'] else 'FAIL':4s}  {c['name']}  ({c['tolerance']})\")

failures = []

def check(name, ok, detail=''):
    print(f\"{'PASS' if ok else 'FAIL':4s}  {name}{'  ' + detail if detail else ''}\")
    if not ok:
        failures.append(name)

print()
print('=== Phase 2: Monte Carlo convergence rates ===')
# Every SE leg must fall as 1/sqrt(N). A slope far from -0.5 means the
# quantity being reported is not a standard error -- which is exactly how the
# degenerate single-fixing control variate showed up as a 1000x reduction.
rows = p2['charts']['mc_error_vs_paths']
n0, n1 = rows[0]['n_paths'], rows[-1]['n_paths']
for leg in ('se_plain', 'se_antithetic', 'se_control_variate', 'se_plain_asian', 'se_sobol'):
    slope = math.log(rows[-1][leg] / rows[0][leg]) / math.log(n1 / n0)
    check(f'mc_slope_{leg}', abs(slope + 0.5) < 0.12, f'slope={slope:+.3f} (want -0.5)')

print()
print('=== Phase 5: case-study invariants ===')

# A buyer hedges by going LONG futures. Its realised hedge payoff must be the
# negative of the seller's, so the two cannot both carry the same sign
# convention. Check each buyer case against the forward move it is built on.
for key, cap_name in (('C_airline', 'Asian call'), ('D_gas_utility', 'Strip of caps')):
    pay = p5[key]['realised_payoffs']
    fwd = list(pay)[0]
    vals_f = pay[fwd]['values']
    vals_c = pay[cap_name]['values']
    # A long cap pays max(F_T - F_t, 0), so it is positive exactly where the
    # long futures leg is positive -- never where it is negative.
    ok = all((c > 0) <= (f > 0) for f, c in zip(vals_f, vals_c))
    check(f'{key}_buyer_hedge_sign', ok, f'cap positive only where long futures is')

# Averaging cannot add variance: the vol of the average must sit below the vol
# of a single fixing, and the Asian must price below the European.
c = p5['C_airline']['trade_off_numbers']
check('C_vol_of_average_below_single', c['vol_reduction_ratio'] < 1.0,
      f\"ratio={c['vol_reduction_ratio']:.3f}\")
check('C_asian_below_european', c['asian_vs_european_premium_pct'] < 0,
      f\"asian vs euro={c['asian_vs_european_premium_pct']:+.1f}%\")

# A knock-out must be cheaper than the vanilla it knocks out of, by an amount
# big enough to be worth quoting. A barrier that saves ~1e-7 is a no-op.
d = p5['D_gas_utility']['trade_off_numbers']
check('D_knockout_is_not_a_noop', d['ko_savings_pct'] > 1.0,
      f\"saving={d['ko_savings_pct']:.2f}% of premium\")
check('D_knockout_touchable', 0.0 < d['ko_realised_hit_frequency'] < 1.0,
      f\"realised hit rate={d['ko_realised_hit_frequency']:.1%}\")

# The seasonal leg must actually differ from the unconditional one, or be
# reported as not differing -- never relabelled as the maturity effect.
check('D_seasonal_is_distinct_from_maturity',
      'seasonal_vs_maturity_pct' in d and 'flat_overprices_vs_maturity_pct' in d,
      f\"flat {d['flat_overprices_vs_maturity_pct']:+.1f}%, seasonal {d['seasonal_vs_maturity_pct']:+.1f}%\")

# Case B: the shortfall is a ZW quote, so it is in cents. Flag anything that
# looks like it has been silently converted to dollars.
b = p5['B_farmer']['trade_off_numbers']
check('B_units_declared', 'cents' in b.get('quote_units', ''), b.get('quote_units', 'MISSING'))
check('B_costly_knockouts_subset', b['n_costly_knockouts'] <= b['n_knockouts'],
      f\"{b['n_costly_knockouts']} costly of {b['n_knockouts']} knock-outs\")

print()
print('=== Phase 6: case-study invariants ===')

# The per-observation autocall frequencies are disjoint events, so they sum to
# the total probability. Dividing that sum by the observation count turns a
# probability into a per-date average -- the bug that made 65% read as 16%.
g = p6['G_structured_note']
mp, hr = g['model_pricing'], g['historical_resampling']
total = sum(mp['autocall_frequency_by_obs'])
check('G_autocall_total_is_a_sum',
      abs(hr['model_expected_autocall_frequency'] - total) < 1e-9,
      f\"reported={hr['model_expected_autocall_frequency']:.4f}, sum={total:.4f}\")
check('G_autocall_complements_never',
      abs(total + mp['never_autocalled_frequency'] - 1.0) < 1e-6,
      f\"sum+never={total + mp['never_autocalled_frequency']:.6f}\")

# The variance ratio must be a ratio of means, not a mean of ratios: a
# near-zero denominator sends the latter to absurd values.
iv = p6['variance_swap_aside']['intraday_vs_closeclose']
implied = iv['intraday_realized_var_mean'] / iv['closeclose_realized_var_mean']
check('varswap_ratio_is_ratio_of_means',
      abs(iv['ratio_intraday_over_closeclose'] - implied) < 1e-6,
      f\"reported={iv['ratio_intraday_over_closeclose']:.3f}, implied={implied:.3f}\")

print()
if failures or not p2['all_passed']:
    print('SOME FAILED: ' + ', '.join(failures) if failures else 'PHASE 2 FAILED')
    sys.exit(1)
print('ALL PASSED')
"
