# Phase 6b prep note: 018's 18-trial family may not be assemblable

Written while Phase 3 runs, independent research per sec 14.1's own instruction
not to compute the corrected DSR number early -- this is about whether the
family CAN be assembled, not what it produces.

## The 12 (unambiguous)

`phase_4_18_results.json` -> `by_offset.{0,1,2,3}.{timed,always_on,cash}.sharpe_net`
(cash is always 0.0). All net, all per-period-comparable after adjusting units
consistently with the headline (`best_sharpe_per_period`). 12 clean values.

## The 6 ablations (NOT unambiguous)

`phase_0_18_preregistration.json`'s n_trials derivation text says:
"6 Phase 5 ablations (no-hysteresis, perp-leg-only, excluding LUNA/FTT, cost
sensitivity at 4 levels counted as configurations, levered 3x/5x, by-year
decomposition) = 18"

But `phase_5_18_results.json`'s actual top-level keys for these 6 "slots":

| slot | key | # of stored sharpe values | net or gross |
|---|---|---|---|
| no-hysteresis | `1_no_hysteresis` | 1 (`sharpe_net`) | net |
| perp-leg-only | `2_perp_leg_only_no_spot_hedge` | 1 (`sharpe` only) | **gross, no sharpe_net field** |
| excl LUNA/FTT | `3_excluding_luna_ftt` | 1 (`sharpe_net`) | net |
| cost sensitivity | `4_cost_sensitivity` | **4** (0.0bp/17.0bp/34.0bp/51.0bp, each `sharpe`) | gross-at-that-cost-level (34bp reproduces the baseline's own sharpe_net exactly) |
| levered | `5_levered_variants` | **2** (3x, 5x, each `sharpe`) | unclear net/gross labeling, no `_net` suffix |
| by-year | `6_by_year_decomposition` | **4** (2021/2022/2023/2024, each `sharpe`) | gross only, no `_net` |

Raw count if every stored value in these 6 slots is taken individually:
1+1+1+4+2+4 = 13, not 6. Total family size would be 12+13=25, not 18.

Also: `2_perp_leg_only`, `4_cost_sensitivity`, `5_levered_variants`, and
`6_by_year_decomposition` have NO `sharpe_net` field at all -- only gross
`sharpe`. Mixing those with the 12 offset values (all `sharpe_net`) would
violate unit consistency (comparable to the annualized-vs-per-period trap in
sec 7.2, but for net-vs-gross). `4_cost_sensitivity`'s `cost_34.0bp` entry's
`sharpe` (0.5766182328943011) is suspicious-looking-net (matches offset_0's
own `sharpe_net` exactly) but that's a coincidence of 018's assumed cost
level equaling the fee assumption used elsewhere, not a general rule for the
other 3 cost levels.

## Conclusion (tentative, to finalize in Phase 6b after the hash gate opens)

There is no non-arbitrary way to pick exactly ONE scalar per "ablation slot"
for the 4 multi-valued slots (cost sensitivity, levered, by-year) without
inventing a selection rule that was never pre-registered -- e.g. "use the
34bp cost level" or "use 2024" would be exactly the kind of after-the-fact
family choice sec 2.2/13.6 warns is undetectable from the output and must be
avoided. Per sec 14.1's mandatory rule ("If the 18 cannot be assembled
exactly, 018's row is reported as not_rescorable -- it does not fall back to
a smaller family"), the likely correct call is: **018's row is
not_rescorable**, and the reason is a genuine, disclosable finding about
018's own record-keeping (parallel to sec 5.4's "future notebooks should
store their trial Sharpe vectors" recommendation) -- the pre-registration's
"6 ablations" language does not correspond to 6 stored scalars.

Do NOT finalize this in the JSON output until Phase 5's hash gate is open
and this is being written as part of Phase 6b proper -- this note is prep,
not the deliverable.
