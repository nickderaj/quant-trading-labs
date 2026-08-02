"""11a Phase 0: reproduction check (NEXT_PROMPT.md sec 3 Phase 0).

Asserts, from our own already-committed JSON, the numbers NEXT_PROMPT.md
sec 0/sec 3 cites as background: Gate SP's per-group Sharpes/DSRs from
phase_1_10b_results.json, the taxonomy/ADF counts from
phase_2_10a_results.json, and the five half-lives in sec 0.2's corroboration
table. Descriptive only -- no new computation, no gate verdict.

Writes phase_0_11a_results.json.
"""

import json

OUT_PATH = "src/research/tmp/phase_0_11a_results.json"

EXPECTED_HALF_LIVES = {
    "brent_calendar": 42.7,
    "brent_wti": 79.3,
    "corn_wheat": 45.4,
    "bean_corn": 118.2,
    "kc_chicago_wheat": 113.4,
}


def close(a: float, b: float, tol: float = 0.15) -> bool:
    return abs(a - b) <= tol


def main() -> None:
    checks: list[dict] = []

    with open("src/research/tmp/phase_1_10b_results.json") as f:
        d10b = json.load(f)
    inter_sharpe = d10b["inter_commodity"]["by_offset"]["offset_0"]["sharpe"]
    cal_sharpe = d10b["calendar"]["by_offset"]["offset_0"]["sharpe"]
    inter_dsr = d10b["inter_commodity"]["deflated_sharpe_prob"]
    cal_dsr = d10b["calendar"]["deflated_sharpe_prob"]
    checks.append(
        {
            "name": "gate_sp_inter_commodity_sharpe",
            "expected": 0.423,
            "actual": inter_sharpe,
            "pass": close(inter_sharpe, 0.423, 0.01),
        }
    )
    checks.append(
        {
            "name": "gate_sp_calendar_sharpe",
            "expected": 0.504,
            "actual": cal_sharpe,
            "pass": close(cal_sharpe, 0.504, 0.01),
        }
    )
    checks.append(
        {
            "name": "gate_sp_inter_commodity_dsr",
            "expected": 0.562,
            "actual": inter_dsr,
            "pass": close(inter_dsr, 0.562, 0.01),
        }
    )
    checks.append(
        {
            "name": "gate_sp_calendar_dsr",
            "expected": 0.680,
            "actual": cal_dsr,
            "pass": close(cal_dsr, 0.680, 0.01),
        }
    )

    with open("src/research/tmp/phase_2_10a_results.json") as f:
        d10a = json.load(f)
    per_spread = d10a["per_spread"]
    n_calendar = sum(1 for v in per_spread.values() if v["taxonomy"] == "calendar")
    n_inter = sum(1 for v in per_spread.values() if v["taxonomy"] == "inter_commodity")
    n_adf_pass = sum(
        1
        for v in per_spread.values()
        if v["adf_cointegration"]["stationary_5pct"] is True
    )
    n_adf_fail = sum(
        1
        for v in per_spread.values()
        if v["adf_cointegration"]["stationary_5pct"] is False
    )
    checks.append(
        {
            "name": "n_spreads_total",
            "expected": 30,
            "actual": len(per_spread),
            "pass": len(per_spread) == 30,
        }
    )
    checks.append(
        {
            "name": "n_calendar",
            "expected": 19,
            "actual": n_calendar,
            "pass": n_calendar == 19,
        }
    )
    checks.append(
        {
            "name": "n_inter_commodity",
            "expected": 11,
            "actual": n_inter,
            "pass": n_inter == 11,
        }
    )
    checks.append(
        {
            "name": "n_adf_pass_5pct",
            "expected": 23,
            "actual": n_adf_pass,
            "pass": n_adf_pass == 23,
        }
    )
    checks.append(
        {
            "name": "n_adf_fail_5pct",
            "expected": 7,
            "actual": n_adf_fail,
            "pass": n_adf_fail == 7,
        }
    )

    half_life_report = {}
    for name, expected in EXPECTED_HALF_LIVES.items():
        actual = per_spread[name]["ar1_mean_reversion"]["half_life_days"]
        ok = close(actual, expected, 0.2)
        half_life_report[name] = {"expected": expected, "actual": actual, "close": ok}
        checks.append(
            {
                "name": f"half_life_{name}",
                "expected": expected,
                "actual": actual,
                "pass": ok,
            }
        )

    all_pass = all(c["pass"] for c in checks)
    out = {
        "checks": checks,
        "all_pass": all_pass,
        "half_life_corroboration": half_life_report,
        "_note": (
            "The taxonomy counts (30/11/19/23) here read as (n_total/n_inter_commodity/"
            "n_calendar/n_adf_pass) -- NEXT_PROMPT.md sec 3 Phase 0 lists them in that order."
        ),
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(
        f"Phase 0: {sum(c['pass'] for c in checks)}/{len(checks)} checks pass, all_pass={all_pass}"
    )


if __name__ == "__main__":
    main()
