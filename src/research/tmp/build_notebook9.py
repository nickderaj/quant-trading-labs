import json


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def code(src):
    return {
        "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
        "source": src.splitlines(keepends=True),
    }


cells = []

cells.append(md("""\
# Notebook 9 - What Are Other People Actually Doing? An External Research Review

Eight notebooks, eight honest nulls on tradeable alpha, alongside real, replicated,
holdout-confirmed results on the risk-modelling side. Rather than testing a ninth
strategy from the same academic playbook, this notebook stops and asks an external
question: what do practitioners, replication literature, and regulators actually report,
and does it look anything like what this programme has been testing?

**This is primarily a literature/practitioner survey, not a modelling notebook.** 36
sources, tiered 1-4 for trust, filed against five competing explanations for the eight
nulls. All four pre-declared gates fire. Full narrative:
`src/results/009_external_research_review.md` (the primary deliverable, more so than this
notebook, per this notebook's own pre-registration). This notebook loads pre-gathered
JSON from `tmp/phase_{0..4}*.json` and narrates/tabulates only - the source record itself
was gathered via WebSearch/WebFetch during construction and is transcribed as data in
`tmp/run_phase_1_survey.py`, exactly as any other phase script writes its own computed
result.
"""))

cells.append(code("""\
import sys

sys.path.insert(0, "..")
sys.path.insert(0, "tmp")
import json

import matplotlib.pyplot as plt
import numpy as np

TMP = "tmp"

def load(name):
    with open(f"{TMP}/{name}") as f:
        return json.load(f)

fig_n = [0]
def show(fig, caption):
    fig_n[0] += 1
    print(f"Figure {fig_n[0]}: {caption}")
    plt.tight_layout()
    plt.show()
"""))

# ---------------------------------------------------------------------------
# Phase 0
# ---------------------------------------------------------------------------
cells.append(md("""\
## Phase 0 - Reproduction check

Re-derives, from the already-committed notebook 7/8 JSONs, the three numbers this
notebook's diagnosis leans on most: Gate CE's 15/16 rejection count, Gate RE's 15/16 pass
count, and - most importantly for hypothesis (c) - Gate AC's exact near-miss shape (net
Sharpe 0.90-0.95 at every offset, deflated Sharpe probability 0.997, excess-vs-basket CI
including zero, does not fire).
"""))

cells.append(code("""\
phase0 = load("phase_0_repro9_results.json")
print("Gate CE repro:", json.dumps(phase0["gate_CE_repro"], indent=2))
print()
print("Gate RE repro:", json.dumps(phase0["gate_RE_repro"], indent=2))
print()
print("Gate AC repro:", json.dumps(phase0["gate_AC_repro"], indent=2))
print()
print(phase0["_verdict"])
"""))

# ---------------------------------------------------------------------------
# Phase 1
# ---------------------------------------------------------------------------
cells.append(md("""\
## Phase 1 - The survey

36 sources, gathered via WebSearch/WebFetch and tiered per this notebook's own
pre-declared source-quality hierarchy (Tier 1 = peer-reviewed replication/regulatory;
Tier 2 = identifiable practitioners discussing costs and failures; Tier 3 = unverified
blog/forum content, hypothesis-generation only; Tier 4 = actively discounted). Every
fetched page was treated as untrusted input throughout - nothing fetched was executed,
no embedded instructions were followed, and no page attempted to direct this notebook's
research process (none triggered the red-flag check for that specifically).
"""))

cells.append(code("""\
phase1 = load("phase_1_survey_results.json")
print(json.dumps(phase1["summary"], indent=2))
"""))

cells.append(md("**tier distribution** across all 36 sources."))
cells.append(code("""\
tiers = phase1["summary"]["tier_counts"]
fig, ax = plt.subplots(figsize=(7, 4.5))
colors = {"1": "#2a9d8f", "2": "#4b3f96", "3": "#d1495b", "4": "#8c1f30"}
labels = [f"Tier {k}" for k in tiers]
ax.bar(labels, tiers.values(), color=[colors[k] for k in tiers])
ax.set_ylabel("number of sources")
ax.set_title("Source tier distribution (36 sources surveyed)")
show(fig, "Tier distribution - roughly half Tier 1 (peer-reviewed/regulatory), a third Tier 2 (identifiable practitioners), the rest Tier 3 (hypothesis-generation only); zero Tier 4 sources were promoted into the evidence base.")
"""))

cells.append(md("**hypothesis coverage matrix** - Tier 1/2 source count per hypothesis (Gate S's own criterion)."))
cells.append(code("""\
hyp_names = {"a": "(a) too naive", "b": "(b) costs too harsh", "c": "(c) bar too strict",
             "d": "(d) markets efficient", "e": "(e) wrong place"}
cov = phase1["summary"]["hypothesis_coverage_tier1_or_2"]
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar([hyp_names[k] for k in cov], cov.values(), color="#2a9d8f")
ax.axhline(5, color="black", lw=1, ls="--", label="Gate S threshold (5)")
ax.set_ylabel("Tier 1/2 sources"); ax.tick_params(axis='x', rotation=20)
ax.legend()
show(fig, "Tier 1/2 source coverage per hypothesis - every hypothesis clears the 5-source Gate S threshold, (d) and (e) most heavily.")
"""))

cells.append(md("**gate S verdict**."))
cells.append(code("""\
print(f\"Gate S fires: {phase1['summary']['gate_S_fires']}\")
print(f\"  {phase1['summary']['n_sources']} sources >= 30: {phase1['summary']['n_sources'] >= 30}\")
print(f\"  min Tier1/2 per hypothesis: {min(cov.values())} >= 5: {min(cov.values()) >= 5}\")
"""))

cells.append(md("**source table** (id, tier, hypotheses addressed) - full detail in the JSON, this is a compact index."))
cells.append(code("""\
for s in phase1["sources"]:
    hyps = ",".join(h["h"] for h in s["hypotheses"])
    print(f\"[T{s['tier']}] {s['id']:42s} hyp={hyps:8s} {s['title'][:70]}\")
"""))

# ---------------------------------------------------------------------------
# Phase 2
# ---------------------------------------------------------------------------
cells.append(md("""\
## Phase 2 - The diagnosis (headline deliverable)

Each hypothesis gets an explicit verdict, with the specific Tier 1/2 sources behind it -
full reasoning text in the JSON and in `src/results/009_external_research_review.md`.
Hypothesis (c) additionally carries a specific, sourced, actionable recommendation on
this repo's own gate criterion, with consequences for the eight existing nulls stated as
an explicitly labelled hypothetical (not a re-score).
"""))

cells.append(code("""\
phase2 = load("phase_2_diagnosis_results.json")
for hkey, h in phase2["hypotheses"].items():
    print(f\"({hkey}) {h['name']}\")
    print(f\"    VERDICT: {h['verdict'].upper()}\")
    print(f\"    supporting: {h['supporting_sources']}\")
    print(f\"    contrary/qualifying: {h['contrary_or_qualifying_sources']}\")
    print()
"""))

cells.append(md("**hypothesis-vs-evidence-tier matrix**."))
cells.append(code("""\
verdict_order = ["well-supported", "partially supported", "contradicted", "insufficient evidence"]
verdict_color = {"well-supported": "#2a9d8f", "partially supported": "#e9c46a",
                  "contradicted": "#d1495b", "insufficient evidence": "#adb5bd"}
fig, ax = plt.subplots(figsize=(9, 4.5))
hkeys = list(phase2["hypotheses"].keys())
verdicts = [phase2["hypotheses"][k]["verdict"] for k in hkeys]
n_sources = [len(phase2["hypotheses"][k]["supporting_sources"]) + len(phase2["hypotheses"][k]["contrary_or_qualifying_sources"]) for k in hkeys]
colors = [verdict_color[v] for v in verdicts]
bars = ax.bar([hyp_names[k] for k in hkeys], n_sources, color=colors)
for b, v in zip(bars, verdicts):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.1, v, ha="center", fontsize=8, rotation=15)
ax.set_ylabel("Tier 1/2 sources cited in verdict"); ax.tick_params(axis='x', rotation=15)
ax.set_title("Diagnosis: verdict and evidence count per hypothesis")
show(fig, "Each hypothesis's verdict (label) and the number of Tier 1/2 sources behind it - (d) and (e) well-supported, (b) contradicted, (a) and (c) partially supported: the survey discriminates, it does not conclude everything is possible.")
"""))

cells.append(md("**Gate DX**."))
cells.append(code("""\
print(json.dumps(phase2["gate_DX"], indent=2))
"""))

cells.append(md("""\
**Hypothesis (c) in full - the bar recommendation.** This is the single highest-value
output of the notebook per its own pre-registration.
"""))

cells.append(code("""\
rec = phase2["hypotheses"]["c"]["actionable_recommendation"]
print("SUMMARY:", rec["summary"])
print()
print("RECOMMENDATION:", rec["recommendation"])
print()
print("CONSEQUENCES (labelled hypothetical, not a re-score):")
print(rec["consequences_for_existing_nulls_labelled_hypothetical"])
"""))

cells.append(md("**Gate BAR**."))
cells.append(code("""\
print(json.dumps(phase2["gate_BAR"], indent=2))
"""))

cells.append(md("**our bar vs. the literature's own Sharpe distribution** - the one chart worth making for hypothesis (c): where does notebook 8's carry near-miss (0.90-0.95) sit relative to disclosed institutional benchmarks found in this survey?"))
cells.append(code("""\
benchmarks = [
    ("Allocator screen-out floor\\n(Tier 3, not counted in verdict)", 0.5),
    ("Man AHL Diversified\\n(disclosed, 1996-2009)", 0.86),
    ("This repo's carry near-miss\\n(notebook 8, Gate AC)", 0.925),
    ("'Good' hedge fund range floor\\n(Tier 3, not counted in verdict)", 1.0),
]
fig, ax = plt.subplots(figsize=(9, 4.5))
names = [b[0] for b in benchmarks]
vals = [b[1] for b in benchmarks]
colors = ["#adb5bd", "#4b3f96", "#8c1f30", "#adb5bd"]
ax.barh(names, vals, color=colors)
ax.set_xlabel("Sharpe ratio")
ax.set_title("Absolute Sharpe context for notebook 8's carry near-miss")
show(fig, "Absolute-Sharpe context (not the criterion that actually failed carry, which was the excess-vs-basket CI - see Phase 2's full reasoning): carry's 0.90-0.95 sits above a real, disclosed institutional trend-following track record (Man AHL, 0.86) and above informal allocator screen-out floors found in this survey.")
"""))

# ---------------------------------------------------------------------------
# Phase 3
# ---------------------------------------------------------------------------
cells.append(md("""\
## Phase 3 - The shortlist

5 candidates, each with the full detail sec 4 Phase 3 requires: mechanism, evidence tier,
data needs, an honest infrastructure assessment, and a pre-registered gate (name, claim,
fire condition) in the exact format notebook 8's sec 5 used. Two candidates (FA, MM) are
explicitly not testable in this repo as-is - included per the pre-registration's own
instruction that this is a finding, not a dead end.
"""))

cells.append(code("""\
phase3 = load("phase_3_shortlist_results.json")
for c in phase3["candidates"]:
    print(f\"{c['gate_name']}: {c['title']}\")
    print(f\"    hypothesis: ({c['hypothesis_addressed']})   evidence tier: {c['evidence_tier']}   testable now: {c['data_already_in_repo']}\")
    print(f\"    fire condition: {c['preregistered_test']['fire_condition']}\")
    print()
print(\"Gate SL:\", json.dumps(phase3[\"gate_SL\"], indent=2))
"""))

cells.append(md("**shortlist testability** - which candidates are buildable with data already in this repo."))
cells.append(code("""\
fig, ax = plt.subplots(figsize=(9, 4.5))
ids = [c["id"] for c in phase3["candidates"]]
testable = [c["data_already_in_repo"] for c in phase3["candidates"]]
colors = ["#2a9d8f" if t else "#d1495b" for t in testable]
ax.bar(ids, [1]*len(ids), color=colors)
ax.set_yticks([])
for i, (gid, t) in enumerate(zip(ids, testable)):
    ax.text(i, 0.5, "testable now" if t else "needs infra\\nnot in repo", ha="center", va="center", fontsize=9, color="white")
ax.set_title("Shortlist: SP/VS/BM testable with existing data; FA/MM need infrastructure this repo lacks")
show(fig, "3 of 5 shortlist candidates are directly testable with data already in this repo; FA needs spot-price data not confirmed present, MM needs L2 order-book/tick data this repo has never had at all.")
"""))

# ---------------------------------------------------------------------------
# Phase 4
# ---------------------------------------------------------------------------
cells.append(md("""\
## Phase 4 - One cheap empirical probe (Gate SP's mechanism, first look only)

Gate SP (structural spread mean-reversion) is the only shortlist candidate meeting all
three of sec 4 Phase 4's criteria: Tier 1 evidence (Gatev-Goetzmann-Rouwenhorst; Zhu
2024), testable with data already in this repo (the 30 pre-built spread series, never
backtested), and cheap (a descriptive AR(1)/IC screen, not a backtest). **This is
explicitly not a gated backtest** - no cost model, no Sharpe, no gate verdict here; only
a sanity check that the mean-reversion mechanism is even present in this repo's own data.
"""))

cells.append(code("""\
phase4 = load("phase_4_spread_probe_results.json")
for name, r in phase4["per_spread"].items():
    if "error" in r:
        print(name, r["error"]); continue
    ar1 = r["ar1_mean_reversion"]; ic = r["zscore_5d_forward_ic"]
    print(f\"{name:20s} beta={ar1['beta']:+.5f} t={ar1['t_stat_beta']:+.2f} half_life={ar1['half_life_days']:.0f}d  IC={ic['ic']:+.3f} p={ic['p_value']:.4f}\")
print()
print(json.dumps(phase4["summary"], indent=2))
"""))

cells.append(md("**AR(1) mean-reversion coefficients** across the 6 spread series probed, with the half-life implied where mean-reverting."))
cells.append(code("""\
names = list(phase4["per_spread"].keys())
betas = [phase4["per_spread"][n]["ar1_mean_reversion"]["beta"] for n in names]
sig = [phase4["per_spread"][n]["ar1_mean_reversion"]["mean_reverting"] for n in names]
colors = ["#2a9d8f" if s else "#adb5bd" for s in sig]
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.bar(names, betas, color=colors)
ax.axhline(0, color="black", lw=0.8)
ax.set_ylabel("AR(1)-in-differences beta"); ax.tick_params(axis='x', rotation=30)
ax.set_title("Spread mean-reversion screen (first look, not a backtest)")
show(fig, "AR(1) mean-reversion coefficients, 6 pre-built commodity spreads - green = flagged mean-reverting (|t|>2); 5 of 6 spreads show a significant negative beta, directionally consistent with the Tier 1 pairs-trading literature. Half-lives of 46-85 days for most series imply low turnover if traded, a separate question from whether it is profitable net of cost - not addressed here, reserved for a properly gated test in notebook 10.")
"""))

# ---------------------------------------------------------------------------
# Bottom line
# ---------------------------------------------------------------------------
cells.append(md("""\
## Bottom line

All four pre-declared gates fire. **S** (36 sources, >=5 Tier-1/2 per hypothesis). **DX**
(hypotheses (d) and (e) well-supported, (b) contradicted, all on Tier 1/2 evidence - the
survey discriminates). **BAR** (a specific, sourced recommendation: do not lower the
deflated-Sharpe threshold, but add a second, prospective "institutionally fundable
absolute performance" flag from notebook 10 onward - consequences for the eight existing
nulls stated as a labelled hypothetical, not a re-score). **SL** (5 shortlist candidates,
3 testable with data already in this repo).

**The diagnosis itself is not the comfortable "we've been doing it wrong" story.**
Hypothesis (d) - markets are efficient at the instruments/horizons this repo can reach -
is well-supported, not the boring fallback option. Hypothesis (e) - we've been looking in
the wrong place - is equally well-supported, but its best-evidenced examples (the
Treasury basis trade, market-making, the volatility risk premium) are almost entirely
walled off from this repo by infrastructure this repo does not have, not by absence of
evidence. The one clear exception - structural mean-reversion in the 30 pre-built
commodity spread series, notebook 8's own declared-and-cut Strategy E - shows a real,
directionally-consistent signal in this notebook's own first-look probe (5/6 spreads
mean-reverting, 4/6 with significant negative IC) and is this notebook's single concrete,
actionable, sourced recommendation for notebook 10.

Full narrative, every source, every reasoning chain: `src/results/009_external_research_review.md`.
"""))

with open("src/research/009_external_research_review.ipynb", "w") as f:
    json.dump(
        {
            "cells": cells,
            "metadata": {
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python", "version": "3.12"},
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        },
        f,
        indent=1,
    )
print(f"written src/research/009_external_research_review.ipynb ({len(cells)} cells)")
