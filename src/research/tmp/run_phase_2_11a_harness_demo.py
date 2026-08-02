"""11a Phase 2: build and validate the evaluation harness (NEXT_PROMPT.md
sec 3 Phase 2) -- `pnl_atr`, `ret_eq`, the paired block bootstrap, and the
noise-floor construction. Descriptive only -- no gate verdict.

Demonstrated here on a synthetic control/treatment pair with a KNOWN
injected effect, to prove the paired construction actually cancels shared
path noise (the reason it's used over an unpaired bootstrap): two synthetic
trade books share the same underlying quarterly "market" shock series, and
the treatment adds a small constant edge on top. The paired CI should
recover that edge far more tightly than either book's own noise floor
would suggest. The REAL noise floor and control-book numbers are reported
in Phase 4 on the actual reproduced control book, not here.

Writes phase_2_11a_results.json.
"""

import json
import sys

sys.path.insert(0, "src/research/tmp")
sys.path.insert(0, "src")

import numpy as np
import spread_lib11 as S11


def main() -> None:
    rng = np.random.default_rng(0)
    n_trades = 400
    n_quarters = 40
    quarter_id = rng.integers(0, n_quarters, size=n_trades)
    quarter_shock = rng.normal(
        0, 500, size=n_quarters
    )  # shared "market" noise per quarter
    idio = rng.normal(0, 200, size=n_trades)
    control_pnl = quarter_shock[quarter_id] + idio
    injected_edge = 15.0
    treatment_pnl = control_pnl + injected_edge + rng.normal(0, 20, size=n_trades)

    dates = np.array(["2015-01-15"], dtype="datetime64[D]") + (quarter_id * 91).astype(
        "timedelta64[D]"
    )
    control_blocks = S11.trade_blocks(dates)
    treatment_blocks = S11.trade_blocks(dates)

    paired = S11.paired_block_bootstrap(
        control_pnl, control_blocks, treatment_pnl, treatment_blocks
    )
    injected_total = (
        injected_edge * n_trades
    )  # paired_block_bootstrap's delta is a SUM, not a per-trade mean

    # For contrast: an UNPAIRED comparison bootstraps each book's own total sum
    # independently (ignoring that both share the same quarter_shock draws),
    # so the shared "market" noise does NOT cancel and the naive delta interval
    # is wider than the paired one for the same underlying data.
    control_floor = S11.noise_floor(control_pnl, control_blocks)
    treatment_floor = S11.noise_floor(treatment_pnl, treatment_blocks)
    naive_delta_lo = treatment_floor["ci_return"][0] - control_floor["ci_return"][1]
    naive_delta_hi = treatment_floor["ci_return"][1] - control_floor["ci_return"][0]

    sample_pnl_atr = S11.pnl_atr(realized_pnl=300.0, quantity=2, atr_at_entry=150.0)
    sample_ret_eq = S11.ret_eq(realized_pnl=300.0, equity_at_open=1_000_000.0)

    out: dict = {
        "synthetic_demo": {
            "n_trades": n_trades,
            "injected_edge_per_trade": injected_edge,
            "injected_edge_total": injected_total,
            "paired_block_bootstrap": paired,
            "paired_ci_recovers_injected_total": bool(
                paired["delta_ci"][0] <= injected_total <= paired["delta_ci"][1]
            ),
            "paired_ci_width": paired["delta_ci"][1] - paired["delta_ci"][0],
            "naive_unpaired_delta_ci": [naive_delta_lo, naive_delta_hi],
            "naive_unpaired_delta_ci_width": naive_delta_hi - naive_delta_lo,
            "pairing_narrows_ci": (paired["delta_ci"][1] - paired["delta_ci"][0])
            < (naive_delta_hi - naive_delta_lo),
        },
        "pnl_atr_example": sample_pnl_atr,
        "ret_eq_example": sample_ret_eq,
        "_note": (
            "The real noise floor and control-book bootstrap are computed on the actual "
            "reproduced control book in Phase 4 (phase_4_11a_results.json), not here -- this "
            "script only proves the harness machinery works as intended, on data where the "
            "true effect (an injected constant per-trade edge, shared quarterly market shock "
            "cancels in the paired construction) is known by construction."
        ),
    }
    with open("src/research/tmp/phase_2_11a_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(
        f"Phase 2: paired delta CI={paired['delta_ci']} (injected total={injected_total}, "
        f"recovered={out['synthetic_demo']['paired_ci_recovers_injected_total']}), "
        f"paired width={out['synthetic_demo']['paired_ci_width']:.1f} vs naive unpaired width="
        f"{out['synthetic_demo']['naive_unpaired_delta_ci_width']:.1f}, "
        f"pairing_narrows={out['synthetic_demo']['pairing_narrows_ci']}"
    )


if __name__ == "__main__":
    main()
