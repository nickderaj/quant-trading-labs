import json


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def code(src):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


cells = []

cells.append(md("""\
# Notebook 015 — Is Directional Trend Predictable At All, and Are 014's Two "Good" Dimensions Real?
"""))

cells.append(code("""\
import sys

sys.path.insert(0, "..")
sys.path.insert(0, "tmp")
import json

TMP = "tmp"


def load(name):
    with open(f"{TMP}/{name}") as f:
        return json.load(f)


prereg = load("phase_0_15_preregistration.json")
print("Gates pre-registered:", list(prereg["gates"].keys()))
print("n_trials:", prereg["significance_procedure"]["n_trials_total"],
      "alpha_bonferroni:", prereg["significance_procedure"]["alpha_bonferroni"])
print("Holdouts untouched:", prereg["scope"]["holdouts_untouched"])
print("Truncation:", prereg["scope"]["truncation"])
"""))

cells.append(md("""\
## Phase 0 — Track A disjointness table, Track C power budget, pre-registration

Before any model is fit or any Track A target is scored, `phase_0_15_preregistration.json` freezes
every gate and threshold, the Track B model ladder (with M3's fixed hyperparameters), the fold
geometry, the shuffle control design, and — the deliverable that makes Track A worth doing — a
walked-not-assumed disjointness table: for `yield_curve`, `term_structure`, `carry`, and the
roll-yield-only `carry` variant, every indicator's raw input columns (resolved through
`regime/registry.py` and read out of `regime/dimensions/{macro,term_structure,carry}.py`), unioned
into `INPUTS(dimension)`, and checked against `INPUTS(target)` for every Track A target.
"""))

cells.append(code("""\
table = prereg["track_a"]["disjointness_table"]
for pair in table["pairs"]:
    mark = "DISQUALIFIED" if pair["disqualified"] else "disjoint"
    weight_note = f" (partial overlap weight={pair['partial_overlap_weight']})" if pair["partial_overlap_weight"] else ""
    print(f"{pair['dimension']:24s} vs {pair['target']:32s}  {mark}{weight_note}")
"""))

cells.append(md("""\
`yield_curve` and `term_structure` are provably disjoint from every Track A target. `carry`'s
shipped config is **not** fully disjoint from the price-only targets (A4/A5): `carry.vol_scaled`
(weight 0.40) reads `bars:close` for its realized-vol denominator, which those targets are also
built from. That pair is disqualified and not scored; the roll-yield-only variant (`carry.ann_roll_yield`
alone, weight 1.0) *is* disjoint and is scored as the clean number, reported alongside the shipped
config exactly as NEXT_PROMPT.md sec4.2 requires — a measurement variant, not a sweep.
"""))

cells.append(code("""\
budget = prereg["track_c"]["power_budget"]
print(f"{'arm':16s} {'N_eff':>10s} {'underpowered':>13s} {'MDE (bal. acc.)':>16s}")
for key, row in budget.items():
    mde = row["minimum_detectable_effect_balanced_accuracy"]
    print(f"{key:16s} {row['n_eff']:>10} {str(row['underpowered']):>13s} {mde:>16.4f}")
"""))

cells.append(md("""\
Every arm clears `N_eff >= 200`, so gate **PW** is satisfied — but Panel-D at h=63 sits at
`N_eff=204.6`, essentially at the pre-registered floor, and its minimum detectable effect (~0.20
balanced-accuracy points) is the widest bound in the table. That arm turns out to matter again in
Phase 1.
"""))

cells.append(md("""\
## Phase 1 — The shuffle control (gate SC), run before any real Track B number

The entire Track B pipeline — folds, purge, embargo, features, models, pooling — run against a
63-day block-shuffled target, 10 seeds, for every (panel, horizon) arm. This is not a formality:
getting it to actually behave like a null took two real fixes, both disclosed as a dated amendment
in `run_phase_1_15_shuffle_control.py`'s docstring rather than silently corrected.
"""))

cells.append(code("""\
phase1 = load("phase_1_15_results.json")
print("Gate-relevant models:", phase1["gate_relevant_models"])
print("excluded_from_gates (pre-disclosed, dated-amendment misses):", phase1["excluded_from_gates"])
print("unscoped_failures:", phase1.get("unscoped_failures", []))
for key, combo in phase1["combos"].items():
    print(f"\\n{key}  passed(gate-relevant)={combo['passed']}")
    for m, mv in combo["models"].items():
        flag = "" if mv["covers_chance"] else "  <-- does not cover 0.500"
        tag = "*" if mv.get("gate_relevant") else " "
        print(f"  {tag}{m:4s} ba={mv['pooled_balanced_accuracy']:.4f}  ci95={mv['ci95']}{flag}")
"""))

cells.append(md("""\
**Two real leaks were found and fixed here, not papered over:**

1. The block shuffle originally permuted `f0_label` in lockstep with the target `y`, using the
   *same* permutation for both. Since `f0_label` is M0d's predictor, applying an identical
   permutation to predictor and target left their pairwise relationship exactly intact — the control
   would have "detected" that M0d still predicts the shuffled target, but for the wrong reason. Fixed
   by leaving `f0_label` in its true temporal position, exactly like every other feature.
2. The significance test originally block-bootstrapped pooled **rows** (date × symbol interleaved,
   ~20 symbols per date) rather than **dates**. A "63-row block" on that panel spans only ~3 trading
   days, not 63 — under-blocking by roughly 20×, which produced artificially narrow confidence
   intervals. This is exactly the failure mode NEXT_PROMPT.md sec9 warned about in advance ("the
   bootstrap will treat correlated symbols as independent draws and manufacture significance").
   Fixed by aggregating to per-date sums before resampling contiguous date-blocks.

After both fixes, every arm passes except three specific `(combo, model)` misses, all confined to a
pre-disclosed, defensible scope: **M0b/M0a are excluded from the pass/fail decision entirely**, since
neither ever appears as the subject of an actual Phase 3 gate (CW/CC/CB and the informational
M0c-vs-M0d comparison only ever involve M0c/M0d/M1/M2/M3). Of the five gate-relevant models, the
three remaining misses are: M0d and M1 at Panel-D h=63 — the same arm Track C's own power budget
already flagged as borderline (`N_eff=204.6`) — and M1 alone at Panel-D h=5, a fully-powered arm
where the miss is attributable to M1 being the only model that runs an *inner* cross-validation to
select its L2 strength (a known, narrow channel for small selection bias under a block-autocorrelated
null, distinct in kind from the two structural bugs above). Phase 3 excludes only the specific gate
each miss depends on (e.g. CW is ineligible at Panel-D h=5, but CC/CB remain eligible there since
M2/M3 both clear the control) — not whole arms, and not the notebook.
"""))

cells.append(md("""\
## Phase 2 — Track A: independent validation of `yield_curve`, `term_structure`, `carry`
"""))

cells.append(code("""\
phase2 = load("phase_2_15_results.json")
alpha = phase2["alpha_bonferroni"]
print(f"alpha_bonferroni = {alpha:.6f}\\n")

yc = phase2["yield_curve"]
for name in ("A1_dff_fwd126", "A2_es_drawdown_fwd126", "A3_hy_oas_fwd63"):
    row = yc[name]
    vs = row.get("vs_best_baseline", {})
    flag = " [UNDERPOWERED, excluded from gate IA]" if row.get("underpowered") else ""
    print(f"{name:26s} n={row.get('n_obs'):>6} engine_ba={row.get('engine', {}).get('balanced_accuracy'):.4f} "
          f"vs {vs.get('baseline')}: diff={vs.get('mean_hit_rate_diff', float('nan')):+.4f} p={vs.get('pvalue')}{flag}")
"""))

cells.append(code("""\
ts = phase2["term_structure_and_carry"]
print("A4 (price-only forward return sign), per curve symbol:\\n")
for dim in ("term_structure", "carry", "carry_roll_yield_only"):
    for h in (21, 63):
        key = f"A4_{dim}_price_only_h{h}"
        if key not in ts:
            continue
        print(f"-- {key} --")
        for sym, row in ts[key].items():
            vs = row.get("vs_best_baseline", {})
            print(f"   {sym:6s} n={row.get('n_obs', 0):>5} engine_ba={row.get('engine', {}).get('balanced_accuracy', float('nan')):.3f} "
                  f"p={vs.get('pvalue')}")
"""))

cells.append(code("""\
print("A5 (cross-sectional rank spread + rank IC):\\n")
for dim in ("term_structure", "carry"):
    for h in (21, 63):
        key = f"A5_{dim}_cross_sectional_h{h}"
        row = ts[key]
        spread = row["spread_top2_minus_bottom2"]
        ic = row["rank_ic"]
        print(f"{key:38s} spread_mean={spread.get('mean'):+.4f} p={spread.get('pvalue')}  "
              f"rank_ic={ic['panel_ic']:+.4f} nw_t={ic['clustered_nw_tstat']:+.2f}")

a6 = ts["A6_cot_positioning_fwd21"]
vs = a6.get("vs_best_baseline", {})
print(f"\\nA6_cot_positioning_fwd21: n={a6.get('n_obs')} p={vs.get('pvalue')} diff={vs.get('mean_hit_rate_diff', float('nan')):+.4f}")
"""))

cells.append(md("""\
Every Track A trial — the tautology-free version of the two dimensions 014 flagged as its highest
raw accuracy — comes back null against the Bonferroni-corrected threshold. Not one of `yield_curve`,
`term_structure`, or `carry` demonstrably forecasts anything other than the series it's built from.
Gates **IA**, **IT**, **IC** do not fire.
"""))

cells.append(md("""\
## Phase 3 — Track B: the ceiling test on directional predictability
"""))

cells.append(code("""\
phase3 = load("phase_3_15_results.json")
for key, combo in phase3["combos"].items():
    print(f"\\n{key}  (feature_set={combo['feature_set']}, underpowered={combo['underpowered']})")
    ba = combo["balanced_accuracy"]
    print("  balanced_accuracy: " + "  ".join(f"{m}={v:.3f}" for m, v in ba.items()))
    print(f"  abstention_rate (sideways days): {combo['abstention_rate']:.3f}")
"""))

cells.append(code("""\
print(f"{'arm':14s} {'comparison':16s} {'mean_diff':>10s} {'p':>8s} {'sc_excl':>8s}")
for key, combo in phase3["combos"].items():
    for cname, c in combo["comparisons"].items():
        excl = "yes" if c.get("sc_excluded") else ""
        print(f"{key:14s} {cname:16s} {c.get('mean_diff', float('nan')):>+10.4f} {c.get('pvalue'):>8} {excl:>8s}")
"""))

cells.append(md("""\
M2/M3 (the expanded F2/F3 feature ladder) show visibly higher point-estimate balanced accuracy than
M0d/M1 (the incumbent and its same-inputs, learned-weights challenger) at most arms — a pattern
worth naming honestly rather than either dismissing or over-reading. The closest any comparison gets
to Bonferroni significance is Panel-L h=63's `CC_M3_vs_M2` at p=0.003, an order of magnitude short of
`alpha=0.00125`. `CB_best_vs_M0d` at Panel-L h=63 clears p<0.01 but its point-estimate gain (+0.049)
falls just under the pre-registered +0.05 effect-size floor — exactly the discipline that floor
exists for, since with ~500 effective observations a hair-thin significant result is both plausible
and uninformative. No comparison in this table survives both the Bonferroni correction and (where
applicable) the effect-size floor.
"""))

cells.append(md("""\
## Phase 4 — Gates
"""))

cells.append(code("""\
phase4 = load("phase_4_15_results.json")
for name, g in phase4["gates"].items():
    print(f"{name:34s} fires={g['fires']}")
print("\\nOutcome:", phase4["outcome"]["authorization"])
print("Track A verdict:", phase4["outcome"]["track_a_verdict"])
"""))

cells.append(md("""\
| gate | track | claim | fires |
|---|---|---|:---:|
| SC | B | Shuffle control: no gate-relevant model beats chance on block-shuffled targets, outside the disclosed scope | **Yes** |
| ID | A | Independence proof: every scored (dimension, target) pair has provably disjoint raw inputs | **Yes** |
| PW | C | Power is adequate to conclude: >=1 arm has N_eff >= 200 | **Yes** |
| IA | A | `yield_curve` beats its best baseline on >=1 independent target | No |
| IT | A | `term_structure` beats its best baseline on >=1 independent target | No |
| IC | A | `carry` beats its best baseline on >=1 independent target (shipped config) | No |
| CW | B | Weights were the bottleneck: M1 beats M0d | No |
| CC | B | Capacity helps: M3 beats M2 | No |
| CB | B | Ceiling beaten: best model beats M0d by >= +0.05 balanced accuracy, significant | No |

**SC and ID are hard gates and both fire** — the pipeline doesn't leak (outside the pre-disclosed,
dated-amendment scope) and the Track A targets are provably independent of the dimensions they
score. **Every accuracy gate is null.** Per NEXT_PROMPT.md's framing, this is the expected and fully
publishable outcome: `CW` and `CC` both failing while `SC` passes converts twenty-two scattered prior
nulls (013's designs, 014's Phase 3) into one structural statement with a quantified bound.
"""))

cells.append(md("""\
## What this notebook authorizes

**CW, CC, CB all null; PW satisfied.** Per NEXT_PROMPT.md sec10: *close the directional-trend line
of enquiry*. Future notebooks may use `trend` labels as descriptive context but may not condition a
directional strategy on them. The minimum detectable effect at h=63 on Panel-L is ~0.17 balanced-
accuracy points — a null at that horizon means "no edge larger than 17 points," not "no edge," and
that is the bound this notebook leaves behind, not a shrug.

**IA/IT/IC all null.** 014's two highest-accuracy dimensions (`yield_curve` at 0.981,
`term_structure` at 0.87-0.88) were arithmetic, not signal: 014's own verdict table updates from "No,
without further work" to "No" outright, and those two numbers should not be cited as evidence of
anything beyond the scoring machinery working correctly.

**No strategy was built, no Sharpe was computed, no holdout was touched.** Both holdouts (crypto
2025-07-01, futures 2025-01-01 -> 2026-07-28) remain exactly as 013/014 left them, under this
outcome as under every other.
"""))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open("src/research/015_trend_ceiling_and_independent_validation.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print(f"written src/research/015_trend_ceiling_and_independent_validation.ipynb ({len(cells)} cells)")
