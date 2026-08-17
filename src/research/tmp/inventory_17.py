"""Notebook 017, Phase 0: the sec 5.5 inventory.

Recursively walks every JSON file under src/research/tmp/ looking for any
key literally named "deflated_sharpe_prob" whose value is a JSON number,
and records (file, json_pointer, value). This is the sweep sec 5.5 claims
finds 73 stored values across 17 files. Run standalone to print the count;
imported by build_phase_0_17_preregistration.py to freeze it.
"""

from __future__ import annotations

import glob
import json
from typing import Any


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _collect_all_numeric_leaves(
    obj: Any, path: str, out: list[tuple[str, float]]
) -> None:
    """Once we're under a key that has declared itself a DSR container
    (e.g. `deflated_sharpe_prob_by_gate`), every numeric leaf beneath it is
    a distinct stored DSR value, whatever its own key is named."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            child_path = f"{path}/{k}"
            if _is_num(v):
                out.append((child_path, float(v)))
            else:
                _collect_all_numeric_leaves(v, child_path, out)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _collect_all_numeric_leaves(v, f"{path}/{i}", out)


def _walk(obj: Any, path: str, out: list[tuple[str, float]], *, exact: bool) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            child_path = f"{path}/{k}"
            is_hit_key = (
                (k == "deflated_sharpe_prob")
                if exact
                else ("deflated_sharpe_prob" in k.lower())
            )
            if is_hit_key:
                if _is_num(v):
                    out.append((child_path, float(v)))
                elif isinstance(v, (dict, list)):
                    # a named container of DSR values (e.g. *_by_gate) -
                    # every numeric leaf inside is a distinct stored value.
                    _collect_all_numeric_leaves(v, child_path, out)
            else:
                _walk(v, child_path, out, exact=exact)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _walk(v, f"{path}/{i}", out, exact=exact)


def scan(
    root: str = "src/research/tmp", *, exact: bool = True
) -> dict[str, list[tuple[str, float]]]:
    """exact=True (default, the sec 5.5 "deflated_sharpe_prob-keyed sweep"):
    only the literal key `deflated_sharpe_prob`. exact=False: also matches
    key-name variants (e.g. `deflated_sharpe_prob_headline`,
    `published_deflated_sharpe_prob`) -- used only for the disclosed
    secondary check, never for the DS-4 gate itself."""
    by_file: dict[str, list[tuple[str, float]]] = {}
    for fp in sorted(glob.glob(f"{root}/**/*.json", recursive=True)):
        with open(fp) as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                continue
        hits: list[tuple[str, float]] = []
        _walk(data, "", hits, exact=exact)
        if hits:
            by_file[fp] = hits
    return by_file


if __name__ == "__main__":
    by_file = scan()
    total = sum(len(v) for v in by_file.values())
    print(f"{total} values across {len(by_file)} files\n")
    for fp, hits in by_file.items():
        print(f"{fp}  ({len(hits)})")
        for ptr, val in hits:
            print(f"    {ptr} = {val}")
