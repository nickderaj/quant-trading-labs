"""Notebook 024 builder -- a narrative over the phase JSONs. Loads results and
renders them; does not re-run any fit.

Usage (from repo root): uv run python src/research/tmp/build_notebook24.py
"""

from __future__ import annotations

import json as _json
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
TMP = REPO_ROOT / "src" / "research" / "tmp"
OUT_PATH = REPO_ROOT / "src" / "research" / "024_samuelson_effect_across_futures.ipynb"


def cid() -> str:
    return uuid.uuid4().hex[:8]


def md(src: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": cid(),
        "metadata": {},
        "source": src.splitlines(keepends=True),
    }


def code(src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cid(),
        "metadata": {},
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


cells: list[dict] = []

# Plain-language explanation for every chart: what it shows, in everyday terms,
# and whether the result matches simple intuition. Kept non-technical on
# purpose -- the numeric detail lives in the write-up and the phase JSONs.
EXPLANATIONS: dict[str, str] = {
    "A1": (
        "**What this shows:** how choppy crude-oil futures prices are, split by how "
        "close each contract is to expiring. The line falls as you move right (further "
        "from expiry), meaning contracts closer to expiring bounce around more from "
        "day to day.\n\n"
        "**Does it match intuition?** Yes. A contract expiring soon settles against "
        "*today's* news — a pipeline outage, an OPEC statement — while a contract "
        "expiring in 8 years barely reacts to this week's headlines. The falling line "
        "is exactly that intuition, drawn from the data."
    ),
    "A2": (
        "**What this shows:** the same falling line as above, but drawn three "
        'different ways of measuring "how much prices moved." The naive method '
        "(plain standard deviation, red) shoots up wildly at the far end, while the "
        "two more careful methods (blue, orange) stay smooth.\n\n"
        "**Does it match intuition?** The smooth decline does; the red spike doesn't "
        "— and shouldn't be trusted. It's the tell that a handful of bad or stale "
        "price prints on old, thinly-traded contracts are corrupting the naive "
        "number, not a real market signal. That's *why* this notebook uses the "
        "sturdier methods everywhere else."
    ),
    "A3": (
        '**What this shows:** the same "calmer further from expiry" picture, '
        "repeated separately for all 16 products in the dataset, so you can eyeball "
        "at a glance which markets behave like crude oil (falling line) and which "
        "don't.\n\n"
        "**Does it match intuition?** Mostly — energy and grains (orange, teal "
        "panels) fall as expected. A few panels, especially the metals (blue), stay "
        "flat or even tick upward, which is the first hint of the inversion this "
        "notebook investigates later."
    ),
    "A4": (
        "**What this shows:** all 16 lines rescaled so every product equals 1.0 at "
        "the one-year mark, making their *shapes* directly comparable regardless of "
        "how volatile each market normally is. A line ending above 1.0 on the right "
        "means that product gets **more** volatile further from expiry — the "
        "opposite of the usual pattern.\n\n"
        "**Does it match intuition?** Most lines dip below 1.0 as expected. The ones "
        "that end up above it are the platinum/palladium/silver group — a genuine "
        "surprise this notebook confirms is not a data glitch (see A6 and the "
        "write-up's metals-inversion section)."
    ),
    "A5": (
        "**What this shows:** the same declining lines, grouped by sector (energy, "
        "metals, grains), with the S&P 500 futures (a plain stock index, dashed "
        'grey) added to every panel as a "should show nothing" baseline.\n\n'
        "**Does it match intuition?** Yes — a stock index has no physical delivery "
        "or storage cost, so there's no obvious reason its volatility should decay "
        "with time-to-expiry the way a real commodity's does. The dashed line stays "
        'close to flat, which is the expected "nothing here" result for a '
        "negative control."
    ),
    "A7": (
        "**What this shows:** a colour-coded table version of A4 — one row per "
        "product, colour showing whether that maturity's volatility is higher (red) "
        "or lower (blue) than that same product's own one-year level. Blank cells "
        "are maturities too thinly traded to trust.\n\n"
        "**Does it match intuition?** The pattern is easy to read at a glance: most "
        "rows go from red (near-term, jumpy) to blue (far-term, calm) left to "
        "right — except the metals rows, which do the opposite."
    ),
    "A6": (
        "**What this shows:** one bar per product, ranking how strongly each one "
        'shows the "calmer further from expiry" pattern, from strongest (top) to '
        "weakest/reversed (bottom). Bars pointing left (negative) match the usual "
        "pattern; bars pointing right (positive) don't. The thin whiskers show how "
        "confident the estimate is — a whisker crossing zero means the result isn't "
        "statistically solid.\n\n"
        "**Does it match intuition?** Mostly. Natural gas and corn lead the pack, "
        "which fits — both have urgent, weather- and harvest-driven near-term news. "
        "The one miss: the S&P 500 (`ES`) was expected to be dead last, since it has "
        "no storage economics at all, but platinum edges it out. `ES` still lands "
        'firmly in the "weak or reversed" group, just not literally last.'
    ),
    "B1": (
        "**What this shows:** crude oil's day-to-day price choppiness over the last "
        "16 years, tracked separately for near-term (blue) and long-term (red) "
        "contracts, with well-known crisis periods shaded (2014 oil crash, COVID, "
        "the Ukraine invasion).\n\n"
        "**Does it match intuition?** Yes — the blue line visibly spikes higher "
        "during every shaded crisis, while the red line barely reacts. That's what "
        '"near-term news matters more up close" should look like in a real, messy '
        "time series."
    ),
    "B2": (
        "**What this shows:** the same information as the chart above, boiled down "
        "to a single number — the near-term choppiness divided by the long-term "
        "choppiness. Above the dashed line (1.0) means near-term contracts are "
        "unusually jumpy relative to long-term ones.\n\n"
        "**Does it match intuition?** Yes — the ratio jumps well above 1.0 during "
        "every shaded crisis window, confirming the effect gets *stronger* exactly "
        "when the news is more urgent, not just present all the time."
    ),
    "B3": (
        "**What this shows:** the same two charts as B1/B2, but for natural gas, "
        "plus a third panel splitting the curve into five separate time-to-expiry "
        "bands instead of just two, to check the effect isn't an artefact of "
        "picking exactly two buckets.\n\n"
        "**Does it match intuition?** Yes, and more so than oil — natural gas is "
        "driven by weather and storage swings that are even more short-lived than "
        "oil's supply/demand news, so a sharper front-end reaction is exactly what "
        "you'd expect."
    ),
    "B4": (
        "**What this shows:** the near-term/long-term ratio from B2, compared "
        "across six very different markets at once — two energy, gold, two grains, "
        "and the S&P 500 — to see whether the pattern is universal or specific to a "
        "handful of commodities.\n\n"
        "**Does it match intuition?** Partially. Oil, gas, corn and wheat all sit "
        "above 1.0 most of the time as expected; gold and the S&P 500 sit close to "
        "1.0, showing much less of the effect — which fits, since neither has real "
        "physical delivery pressure."
    ),
    "B5": (
        "**What this shows:** the *entire* volatility curve (not just a single "
        "ratio) on a handful of specific real dates — some calm, some mid-crisis — "
        "to see how the whole shape shifts, not just the front-to-back gap.\n\n"
        "**Does it match intuition?** Yes — the crisis-date curves (red/orange) sit "
        "well above the calm-date curves (grey) at the front of the curve, but the "
        "gap narrows sharply by the time you reach the far end, which is the "
        "maturity effect playing out on real historical days rather than an average."
    ),
    "B6": (
        "**What this shows:** not volatility itself, but how *unpredictable the "
        "volatility number is* — averaged by sector, for each maturity band. A "
        "tall bar means that part of the curve's riskiness is itself hard to "
        "forecast, even day to day.\n\n"
        "**Does it match intuition?** Roughly — energy's near-term vol-of-vol is "
        "the highest of any sector, matching how often oil and gas get hit by "
        "sudden supply shocks; grains and metals are calmer and more stable "
        "throughout."
    ),
    "B7": (
        "**What this shows:** how closely a market's near-term and long-term "
        "contracts move together on the same day, tracked over time. When this "
        "line drops toward zero, the front and back of the curve are reacting to "
        "completely different things that day.\n\n"
        "**Does it match intuition?** Yes — the correlation dips sharply at times "
        "(down to essentially zero at its lowest point), which is the flip side of "
        "the Samuelson effect: if the front reacts to news the back ignores, the "
        "two should occasionally stop moving together entirely."
    ),
    "B8": (
        "**What this shows:** the same curve shape as the very first chart (A1), "
        "drawn separately for every calendar year from 2010 to 2026, to check "
        'whether "calmer further from expiry" is a stable, every-year feature or '
        "something that only shows up occasionally.\n\n"
        "**Does it match intuition?** Yes, it's remarkably stable — nearly every "
        "year's panel shows the same downward-sloping shape, including calm years "
        "and crisis years alike, which is reassuring: this isn't a fluke of one "
        "unusual year."
    ),
    "C1": (
        "**What this shows:** natural gas's volatility curve split by delivery "
        "season — contracts that deliver gas in winter (Dec-Mar, heating season) "
        "versus summer (Apr-Oct).\n\n"
        "**Does it match intuition?** Yes — the winter line sits clearly above the "
        "summer line at the near-term end. A cold snap is urgent, market-moving "
        "news for a January-delivery contract in a way it simply isn't for a "
        "July-delivery one."
    ),
    "C2": (
        "**What this shows:** volatility by delivery month for four grain crops "
        "(corn, soybeans, wheat, KC wheat), lined up against the farming calendar — "
        "planting, growing season, and harvest.\n\n"
        "**Does it match intuition?** Broadly yes — the months around planting "
        "decisions and harvest uncertainty (when weather can still make or break "
        "the crop) tend to show higher bars than the deep off-season months, "
        "though the pattern is noisier than natural gas's clean winter/summer split."
    ),
    "C3": (
        "**What this shows:** a calendar-style heatmap for natural gas and crude "
        "oil, cross-referencing month-of-year against how far out the contract is, "
        "to spot seasonal hot spots at a glance.\n\n"
        "**Does it match intuition?** For natural gas, yes — the darkest cells "
        "cluster in the winter months at the near-term end, exactly where the C1 "
        "chart already pointed. Crude's pattern is flatter, matching its weaker "
        "seasonal story."
    ),
    "C4": (
        "**What this shows:** the same winter-vs-summer test as C1, applied to "
        "gold instead of natural gas.\n\n"
        "**Does it match intuition?** Yes — and the *lack* of a gap here is the "
        "point. Nobody needs gold delivered in January to heat their home, so gold "
        "has no real delivery season. The two lines sitting almost on top of each "
        'other is the expected "nothing here" result, and it\'s reassuring evidence '
        "that C1's winter/summer gap for gas reflects a real seasonal effect rather "
        "than an artefact of the method."
    ),
    "D1": (
        "**What this shows:** how lopsided each product's daily price moves are at "
        "each maturity — a negative number means big down-days are more extreme or "
        "more common than big up-days, and vice versa.\n\n"
        "**Does it match intuition?** Partially — energy products (orange) tend "
        "toward negative skew at the front of the curve, consistent with how "
        "commodity crashes (like the 2020 oil collapse) tend to be sharper than "
        "commodity rallies. The pattern is noisier further from expiry, where "
        "there's simply less data to pin down a shape confidently."
    ),
    "D2": (
        '**What this shows:** how "fat-tailed" daily returns are at each '
        "maturity — how much more often extreme, surprising moves happen than a "
        "smooth bell-curve would predict. Higher means more shock-prone.\n\n"
        "**Does it match intuition?** Yes — every product is fat-tailed to some "
        "degree (real markets always are), and the near-term end tends to be the "
        "most extreme, which lines up with the main finding: the front of the "
        "curve is where the biggest surprises land."
    ),
    "D3": (
        "**What this shows:** the centrepiece of this section. Instead of plotting "
        "volatility against time-to-expiry, this plots it against how far a "
        "contract's price sits from the cheapest (front) contract that day. This is "
        "built from real settlement prices, not options, so it's labelled "
        '"realised," not "implied."\n\n'
        "**Does it match intuition?** Partly. Crude oil is *not* a symmetric U — it "
        "dips slightly on the discounted (backwardated) side, then climbs steadily "
        "the further a contract trades at a premium (contango), so volatility is "
        "highest for contracts priced richest relative to the front, not "
        "symmetrically at both extremes. That shape survives even after controlling "
        "for time-to-expiry separately, so it's genuine, not the maturity effect in "
        "disguise. Natural gas looks different again — a real peaked hump centred "
        "near the front contract, falling off on both sides — which the write-up "
        "reports honestly rather than forcing it to match oil's shape."
    ),
    "D4": (
        "**What this shows:** the volatility-vs-time-to-expiry curve split "
        "depending on whether the market is currently short on supply "
        "(backwardation — near-term price higher) or oversupplied (contango).\n\n"
        "**Does it match intuition?** For crude oil, yes — backwardated curves "
        "(scarcity) decline noticeably more steeply than contango curves, matching "
        "the idea that a shortage makes near-term news more urgent. For natural gas "
        "the two are essentially identical, so the effect isn't universal. (An "
        "earlier pass at this chart found the opposite for oil; that traced back to "
        "the same far-dated data-quality bug fixed elsewhere in this notebook — see "
        "the write-up's Bugs found section.)"
    ),
    "D5": (
        "**What this shows:** the full shape of daily price moves for near-term "
        "vs. long-term contracts side by side, not just their average size — so "
        "you can see how often truly extreme days happen, not only how volatile "
        "things are on a typical day.\n\n"
        "**Does it match intuition?** Yes — the near-term (blue) distribution is "
        "visibly wider and has fatter tails than the long-term (red) one, "
        "consistent with everything else in this notebook: the front of the curve "
        "doesn't just move more on average, it's also more prone to the rare, "
        "extreme day."
    ),
}


def explain(fig_id: str) -> dict:
    return md(EXPLANATIONS[fig_id])


cells.append(
    md("""\
# Notebook 024 — The Samuelson effect across 16 futures markets

The Samuelson (1965) hypothesis says futures return volatility rises as a contract
approaches expiry: information about near-term supply and demand decays by the time a
deferred contract expires, so the front of the curve reacts to news the back barely
feels. This notebook is a **descriptive, visual atlas** of that effect across every
futures market in this repo (16 products, 2010-2026) -- it fits no trading rule,
computes no Sharpe ratio, and spends no holdout.

Full spec: `NEXT_PROMPT.md`. Full narrative and numbers:
`src/results/024_samuelson_effect_across_futures.md`.
""")
)

cells.append(
    code("""\
import json

TMP = "tmp"


def load(name):
    with open(f"{TMP}/{name}") as f:
        return json.load(f)


prereg = load("phase_0_24_preregistration.json")
prereg["statement"]
""")
)

cells.append(
    md(
        "## Phase 0 — Pre-registration\n\n"
        "This is a descriptive study: no holdout spent, no trading gate. The "
        "falsification condition, the estimator hierarchy, and the hygiene rules "
        "were fixed before any chart was looked at."
    )
)
cells.append(code('prereg["falsification_condition"]'))
cells.append(code('prereg["predicted_ordering_confirmed_scoping_expectation"]'))

cells.append(
    md(
        "## Phase 1 — Panel construction and data-quality census\n\n"
        "845,839 usable rows across 16 products after dropping `close <= 0` and "
        "excluding every contract that ever printed a non-positive close (see "
        "Bugs found in the write-up) "
        "(this is the single most important section of the underlying "
        "prompt -- the far-dated end of every panel contains junk prints that "
        "destroy a plain standard deviation; see Phase 2 for the MAD-vs-std "
        "comparison)."
    )
)
cells.append(
    code("""\
phase1 = load("phase_1_24_panel_report.json")
for p in ["CL", "NG", "ES", "GC"]:
    d = phase1["products"][p]
    print(f"{p}: raw_rows={d['raw_rows']}, close<=0={d['close_le_0_count']}, "
          f"usable(gap1,vol0)={d['usable_returns_variants']['gap1_vol0']['n']}, "
          f"junk(|r|>1)={d['junk_prints_gt_1_log_return']['count']}")
""")
)
cells.append(
    md(
        "These four match the smoke-test numbers in `NEXT_PROMPT.md` section 1 exactly "
        "(CL usable=81,895; NG=93,612; ES=6,797; GC=38,484) -- the pipeline is verified "
        "correct before any chart is built."
    )
)

cells.append(
    code("""\
import sys

sys.path.insert(0, "tmp")
import matplotlib.pyplot as plt
import viz24 as vz

plt.rcParams["figure.facecolor"] = vz.SURFACE
plt.rcParams["savefig.facecolor"] = vz.SURFACE

phase2 = load("phase_2_24_profiles.json")
phase3 = load("phase_3_24_slopes.json")
phase4 = load("phase_4_24_timeseries.json")
phase5 = load("phase_5_24_seasonality.json")
phase6 = load("phase_6_24_smile_skew.json")
""")
)

cells.append(
    md(
        "## Section A — the effect exists\n\n"
        "Seven figures over Phase 2's vol-vs-maturity profiles (16 products x 3 "
        "estimators x 3 liquidity variants x 2 gap variants) and Phase 3's Samuelson "
        "slopes."
    )
)

section_a = [
    (
        "A1",
        "CL, the hero chart: robust vol vs. days-to-expiry with a fitted power law.",
    ),
    (
        "A2",
        (
            "CL, three estimators overlaid -- the std line detonating at the long end "
            "*is* the data-quality finding from section 2 of the prompt."
        ),
    ),
    (
        "A3",
        "All 16 products, small multiples, shared log-x, panel borders tinted by sector.",
    ),
    (
        "A4",
        (
            "All 16 overlaid, each normalised to its own 365-day vol -- the metals "
            "crossing above 1.0 at the long end is the inversion, visible."
        ),
    ),
    (
        "A5",
        (
            "Energy / metals / ags sector panels, with ES as a dashed grey reference "
            "in every one."
        ),
    ),
    (
        "A7",
        "Heatmap: product x maturity bucket, normalised to each product's 365-day vol.",
    ),
]
for fig_id, caption in section_a:
    cells.append(md(f"**{fig_id}.** {caption}"))
    cells.append(
        code(f"""\
import charts_24_a as ca

fig, ax = ca.fig_{fig_id}({{"profiles": phase2, "slopes": phase3}})
plt.show()
""")
    )
    cells.append(explain(fig_id))

cells.append(
    md(
        "## Phase 3 — Samuelson slopes and the metals inversion\n\n"
        "Ranked slopes (mad estimator, `max_gap=1`, `min_volume=0`), with "
        "contract-level bootstrap CIs."
    )
)
cells.append(
    code("""\
for r in phase3["ranking_primary"]:
    print(f"{r['product']:4s} slope={r['slope']:+.4f}  ci=[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]")
""")
)
cells.append(
    md(
        "**Finding that deviates from the scoping expectation:** `ES` was expected to "
        "be the single weakest (most positive) slope of all 16 products -- it is not. "
        "`PL` is slightly more positive than `ES`, and `PA`/`SI` sit close behind. `ES` "
        "does land firmly inside the near-zero/positive cluster, well separated from "
        "the strongly negative energy/ags group, which is the qualitative claim that "
        'matters -- but the precise "weakest of all" ranking was not exactly right, '
        "and the write-up says so."
    )
)
cells.append(md("**A6.** Ranked Samuelson slopes with bootstrap CIs, by sector."))
cells.append(
    code("""\
fig, ax = ca.fig_A6({"profiles": phase2, "slopes": phase3})
plt.show()
""")
)
cells.append(explain("A6"))
cells.append(
    md(
        "**The metals-inversion investigation.** `PL` and `PA` show a positive "
        "(inverted) Samuelson slope. Does it survive tighter liquidity and a "
        "contract-count floor, or is it a handful of thin far-dated contracts?"
    )
)
cells.append(
    code("""\
for p in ["PL", "PA"]:
    inv = phase3["metals_inversion"][p]
    print(p, "verdict:", inv["verdict"])
    print("  primary slope:", round(inv["primary"]["slope"], 4),
          " vol>1000:", round(inv["vol1000"]["slope"], 4),
          " gap<=3:", round(inv["gap3"]["slope"], 4))
    print("  365-730d bucket:", inv["bucket_365_730_stats"])
phase3["metals_inversion_contract_floor"]
""")
)

cells.append(
    md(
        "## Section B — how short-dated vol moves against long-dated vol over time\n\n"
        "Eight figures over Phase 4's 63-day causal rolling vol, crisis-vs-calm "
        "windows, and the front/deferred return correlation."
    )
)
section_b = [
    (
        "B1",
        (
            "CL: rolling front vs. deferred vol, 2010-2026, crisis windows shaded -- "
            "the chart that answers the notebook's core question."
        ),
    ),
    ("B2", "CL: the front/deferred ratio as its own series."),
    ("B3", "The same two-panel treatment for NG."),
    ("B4", "Six products' front/deferred ratio on one axis."),
    (
        "B5",
        (
            "Term structure on selected calm and crisis dates (via nearest annual "
            "aggregate -- see the code comment)."
        ),
    ),
    ("B6", "Vol-of-vol by maturity bucket, sector means."),
    (
        "B7",
        (
            "Rolling 126-day correlation between front and deferred daily returns, "
            "CL and NG -- the de-correlation companion effect."
        ),
    ),
    ("B8", "CL annual small multiples, 2010-2026: is the effect stable or episodic?"),
]
for fig_id, caption in section_b:
    cells.append(md(f"**{fig_id}.** {caption}"))
    cells.append(
        code(f"""\
import charts_24_b as cb

fig, ax = cb.fig_{fig_id}({{"timeseries": phase4}})
plt.show()
""")
    )
    cells.append(explain(fig_id))
cells.append(
    code("""\
for name, stats in phase4["products"]["CL"]["crisis_vs_calm"].items():
    print(name, {k: round(v, 3) for k, v in stats.items()})
""")
)
cells.append(
    md(
        "Front-end vol spikes materially more than deferred vol in every crisis "
        "window relative to the 2017 calm control, and the CL/NG front-deferred "
        "return correlation drops sharply in at least one window (see the printed "
        "`corr_min` above B7) -- the de-correlation companion to the Samuelson effect."
    )
)

cells.append(
    md(
        "## Section C — seasonality\n\n"
        "Four figures over Phase 5: NG's winter/summer split, the grains against "
        "the harvest calendar, calendar-month heatmaps, and GC as the flat control."
    )
)
section_c = [
    ("C1", "NG: winter-delivery vs. summer-delivery contracts."),
    ("C2", "Grains (ZC/ZS/ZW/KE) vol by delivery month."),
    ("C3", "Calendar month x maturity-bucket heatmap, NG and CL."),
    ("C4", 'GC as the flat control -- the "nothing here" chart, which is the point.'),
]
for fig_id, caption in section_c:
    cells.append(md(f"**{fig_id}.** {caption}"))
    cells.append(
        code(f"""\
import charts_24_c as cc

fig, ax = cc.fig_{fig_id}({{"seasonality": phase5}})
plt.show()
""")
    )
    cells.append(explain(fig_id))
cells.append(
    code("""\
ng_winter = phase5["ng_seasonality"]["winter"][0]["vol"]
ng_summer = phase5["ng_seasonality"]["summer"][0]["vol"]
gc_winter = phase5["gc_control"]["winter"][0]["vol"]
gc_summer = phase5["gc_control"]["summer"][0]["vol"]
print(f"NG front-end vol: winter={ng_winter:.3f} summer={ng_summer:.3f} "
      f"ratio={ng_winter / ng_summer:.2f}")
print(f"GC front-end vol: winter={gc_winter:.3f} summer={gc_summer:.3f} "
      f"ratio={gc_winter / gc_summer:.2f}")
""")
)

cells.append(
    md(
        "## Section D — the realised smile/skew exploration\n\n"
        "**Every chart below is realised, not implied.** There are no options in "
        "this repo, so a true implied-vol smile is impossible -- these are realised "
        "analogues built from settlement prices, and are labelled that way "
        "everywhere. Five figures over Phase 6."
    )
)
section_d = [
    ("D1", "Realised skewness by maturity bucket, all 16 products, sector-coloured."),
    ("D2", "Realised excess kurtosis by maturity bucket, log-y."),
    (
        "D3",
        (
            'The realised "smile": vol vs. log-moneyness, pooled and by maturity bin '
            "(the maturity/moneyness confound control)."
        ),
    ),
    ("D4", "Vol-vs-dte split by curve state: contango vs. backwardation, CL and NG."),
    ("D5", "Front vs. deferred return densities, CL and NG, log-y."),
]
for fig_id, caption in section_d:
    cells.append(md(f"**{fig_id}.** {caption}"))
    cells.append(
        code(f"""\
import charts_24_d as cd

fig, ax = cd.fig_{fig_id}({{"smile_skew": phase6}})
plt.show()
""")
    )
    cells.append(explain(fig_id))
cells.append(
    md(
        "**Confound check:** CL's pooled realised smile is not a symmetric U -- it "
        "dips shallowly on the backwardated side then climbs steadily through "
        "contango -- but that asymmetric shape persists within each maturity bin, "
        "so it is not just the Samuelson effect in disguise. NG's pooled shape is "
        "genuinely different: a peaked hump centred near the front contract, also "
        "persisting once maturity is controlled for. The two products do not tell "
        "the same story, and neither is forced into matching the other.\n\n"
        "**Curve-state check:** the naive expectation was that backwardated curves "
        "(signalling scarcity) would show a steeper Samuelson slope than contango "
        "curves. For CL, backwardation *is* markedly steeper (-0.218 vs. -0.149), "
        "matching that intuition. (An earlier pass at this check found the opposite "
        "sign for CL; that was traced to the far-dated data-quality bug documented "
        "in the write-up's Bugs found section and is superseded by the figure here.) "
        "For NG the two are nearly identical (-0.284 vs. -0.292)."
    )
)

cells.append(
    md(
        "## Bottom line\n\n"
        "See `src/results/024_samuelson_effect_across_futures.md` for the full "
        "write-up."
    )
)

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT_PATH.write_text(_json.dumps(nb, indent=1))
print(f"wrote {OUT_PATH}")
