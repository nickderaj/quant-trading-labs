"""Run against the SOURCE engine (../ultron/libs/finance's ultron_finance,
via its own uv venv) to produce the port-fidelity baseline for notebook 014
Phase 0. Invoked as:

    uv run --project ../ultron/libs/finance python3 \
        src/research/tmp/fidelity_source_14.py <out.json>

Writes scores + labels (as JSON-serializable records) for the macro_default
and commodity_default synthetic fixtures.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fidelity_fixture_14 import commodity_inputs, macro_inputs
from ultron_finance.regime import (  # type: ignore[import-not-found]
    RegimeEngine,
    RegimeInputs,
)


def _frame_to_records(frame):
    out = frame.copy()
    out.index = out.index.strftime("%Y-%m-%d")
    return json.loads(out.to_json(orient="index"))


def main() -> None:
    out_path = sys.argv[1]

    macro_ohlcv, macro_macro, macro_cot = macro_inputs()
    macro_result = RegimeEngine.from_default("macro_default").detect(
        RegimeInputs(ohlcv=macro_ohlcv, macro=macro_macro, cot=macro_cot)
    )

    commodity_ohlcv, commodity_curve = commodity_inputs()
    commodity_result = RegimeEngine.from_default("commodity_default").detect(
        RegimeInputs(ohlcv=commodity_ohlcv, curve=commodity_curve)
    )

    payload = {
        "macro_default": {
            "scores": _frame_to_records(macro_result.scores),
            "labels": _frame_to_records(macro_result.labels.astype(object)),
            "config_hash": macro_result.config.config_hash(),
        },
        "commodity_default": {
            "scores": _frame_to_records(commodity_result.scores),
            "labels": _frame_to_records(commodity_result.labels.astype(object)),
            "config_hash": commodity_result.config.config_hash(),
        },
    }
    with open(out_path, "w") as f:
        json.dump(payload, f)


if __name__ == "__main__":
    main()
