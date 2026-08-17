# Notebook 017 — kickoff prompt

Paste into a fresh Claude Code session at the repo root.

---

Read `NEXT_PROMPT.md` in full before doing anything, then execute notebook 017 end to end. It is
self-contained: no market data, no downloads, no model fitting. Inputs are synthetic Monte Carlo
draws and JSON already on disk.

**Order is load-bearing. Work Phase 0 → 1 → 2 → 3 → 4 → 5 → 6 → 6b and do not reorder or overlap.**

- **Phase 0** freezes the pre-registration, including all three §14.2 branch texts, before Phase 2
  runs. Reproduce the 73-value inventory (§5.5) and verify 018's `fires_except_dsr_leg`,
  `bootstrap_ci_leg_fires` and `holdout_access` fields from the stored JSON while you are there.
- **Phase 2 is a kill switch.** If Gate DS-1 does not fire, stop, leave `src/research.py` untouched,
  write up the null per §5.1, and skip Phases 3–5. Treat that as a real outcome, not a formality.
- **Phase 3 is the only expensive step.** Launch `scripts/run_dsr_calibration.sh` with Bash
  `run_in_background: true` and then stop working on it. Do not tail the log, do not poll in a loop,
  do not re-run a finished cell to check. Wire it with `--smoke` first (M=500, ~1 min) to prove the
  plumbing, then launch the real grid once.
- While Phase 3 runs, either idle or do strictly independent work (the notebook scaffold, the
  write-up skeleton). If you want progress, use `/loop 1800s` with a Haiku subagent whose whole
  prompt is: *"read scratch/017/status.json and reply with one line: state, done/total, eta."*
  Nothing more — no log reading, no diagnosis.
- **Phase 5 will refuse to run** unless the §10 hash gate matches. That is intended. If it trips, do
  not bypass it — re-stamp the certificate through Phase 4 as §10 describes.
- **Phase 6b** applies the §14 amendment to 018 by selecting a frozen branch text. Read §14.1 first:
  do not assume 018's corrected DSR leaps, and assemble its trial family from all 18 declared trials
  across both JSONs or mark the row `not_rescorable`.

Constraints I care about:

- Never mutate `phase_4_18_results.json` or `phase_5_18_results.json`, and never re-execute 018's
  notebook. §14.3.
- `src/research.py` gets exactly one function changed, under §8's backward-compatibility contract.
  `run_phase_0_repro.py`'s 0.997 assertion must still pass untouched.
- CI green before any push, per `CLAUDE.md`: `uv run ruff check .`, `uv run ruff format --check .`,
  `uv run mypy .`, `uv run pytest tests/ -q`.
- Commit per phase, as 018 did.

If a gate threshold turns out to be badly chosen, report the measured number against the frozen
threshold anyway and say so in the write-up. Do not retune it. §12.3.

Report back when Phase 2 resolves DS-1 (that decides whether the rest of the run happens), and again
when everything is done.
