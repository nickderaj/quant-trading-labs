"""Phase 0 -- pre-registration for notebook 024 (the Samuelson effect across 16
futures markets). Written before any chart is looked at.

This is a DESCRIPTIVE study: no trading rule, no Sharpe ratio, no P&L, and no
holdout is spent. There is no train/test split and no pre-registered pass/fail
gate on a trading claim. See NEXT_PROMPT.md for the full spec this implements.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib24

TMP = Path(__file__).resolve().parent

PREREG = {
    "written": "2026-09-22",
    "study_type": "descriptive",
    "no_holdout_spent": True,
    "no_trading_gate": True,
    "statement": (
        "This notebook is a descriptive, visual atlas of the Samuelson (1965) "
        "maturity effect across all 16 futures markets in this repo. It fits no "
        "trading rule, computes no Sharpe ratio, and spends no holdout -- the "
        "full sample is used throughout, and there is no train/test split. "
        "Every number is a full-sample descriptive statistic."
    ),
    "falsification_condition": (
        "The study's central claim -- that realised volatility rises as a "
        "contract approaches expiry, and that this effect is strongest in "
        "storable commodities and weakest or absent in financial/precious-metal "
        "instruments -- would be FALSE if: (a) samuelson_slope is not reliably "
        "negative (with a bootstrap 95% CI excluding zero) in the energy and ag "
        "products under the primary MAD estimator; (b) the sign of the slope "
        "flips between the MAD and winsorised-std estimators for any product "
        "where the primary estimator finds a slope with a CI excluding zero; or "
        "(c) the sign flips across the max_gap in {1,3} or min_volume in "
        "{0,100,1000} liquidity variants for a headline product (CL, NG, GC, ES)."
    ),
    "predicted_ordering_confirmed_scoping_expectation": (
        "Energy and grains strongly negative slope (vol rises toward expiry); "
        "precious metals (GC, SI) near zero or weakly positive; ES (negative "
        "control, no storage/convenience-yield economics) weakest of all "
        "commodities. This was already observed during scoping on 2026-09-22 "
        "(see NEXT_PROMPT.md section 3) using the exact method below -- it is "
        "recorded here as a CONFIRMED SCOPING EXPECTATION, not a blind "
        "pre-registered prediction, and the write-up says so explicitly. No "
        "pre-registration credit is claimed for having predicted something "
        "already measured."
    ),
    "estimator_hierarchy": {
        "primary": "mad_vol: 1.4826 * median(|r - median(r)|) * sqrt(252)",
        "secondary_contamination_check": "std_vol: plain sample std, annualised",
        "secondary_robustness_check": (
            "winsor_vol: symmetric 1%-winsorised std, annualised"
        ),
        "rule": (
            "MAD is the headline number in every chart and table. std and "
            "winsor are shown alongside it in at least one chart each to "
            "document contamination and confirm the conclusion is not an "
            "artefact of the specific robust estimator chosen."
        ),
    },
    "hygiene_rules_fixed_in_advance": [
        "Drop rows with close <= 0 (ln undefined); count per product.",
        (
            f"Except: {lib24.NEGATIVE_PRICE_EVENT['ticker']} settled at "
            f"{lib24.NEGATIVE_PRICE_EVENT['close']} on "
            "2020-04-20 is genuine market "
            "history (negative WTI), excluded for the mechanical reason that "
            "ln() is undefined, not treated as junk."
        ),
        "Returns computed within contract_id only, never chained across a roll.",
        (
            "Consecutive-day filter: keep a return only when the calendar gap "
            "to the previous observation of the same contract is exactly 1 "
            "day (primary); gap<=3 is run as a sensitivity variant in Phase 2."
        ),
        "dte = (expiry - date).days; verified non-negative.",
        (
            "Liquidity variants carried through Phase 2 as a sensitivity: "
            "all, volume>100, volume>1000. 'all' is primary for headline "
            "charts because the robust estimator already handles the junk."
        ),
    ],
    "products": lib24.PRODUCTS,
    "sector": lib24.SECTOR,
    "negative_control": "ES",
    "soft_controls": ["GC", "SI"],
    "dte_bins": lib24.DTE_BINS,
    "crisis_windows": lib24.CRISIS_WINDOWS,
}

if __name__ == "__main__":
    out = TMP / "phase_0_24_preregistration.json"
    out.write_text(json.dumps(PREREG, indent=2, default=str))
    print(f"wrote {out}")
