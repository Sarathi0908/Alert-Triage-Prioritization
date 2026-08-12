"""
pipeline.py  —  end-to-end Alert Triage build (#01, CSE-CIC-IDS2018 variant).

Runs the whole chain in dependency order:

  screen    src/features/feature_screener.py   native columns -> data/native_selected.json
  train     src/models/lightgbm_triage.py      the model -> outputs/lgbm_model.pkl + metrics.json
  eda       src/evaluation/eda_report.py       outputs/eda/*  (needs the screened feature set)
  excels    src/reporting/build_excels.py      data/*.xlsx feature sheets
  notebook  src/reporting/make_notebook.py     notebooks/01_eda_cicids2017.ipynb
  postman   src/reporting/build_collection.py  postman/*.json (needs the trained bundle)
  deck      src/reporting/build_deck.py        outputs/*.pptx (needs metrics + EDA)

Each step runs in its own interpreter so one failing report cannot corrupt the
run that produced the model. All settings come from configs/*.yaml.

Run:  py src/pipeline.py                       # everything
      py src/pipeline.py --steps train eda     # a subset, in the given order
      py src/pipeline.py --skip screen deck    # everything except these
      py src/pipeline.py --list                # show the steps and exit
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

_SLOT_ROOT = Path(__file__).resolve().parents[1]
if str(_SLOT_ROOT) not in sys.path:
    sys.path.insert(0, str(_SLOT_ROOT))

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

# step name -> slot-relative script, in dependency order.
STEPS: dict[str, str] = {
    "screen": "src/features/feature_screener.py",
    "train": "src/models/lightgbm_triage.py",
    "eda": "src/evaluation/eda_report.py",
    "excels": "src/reporting/build_excels.py",
    "notebook": "src/reporting/make_notebook.py",
    "postman": "src/reporting/build_collection.py",
    "deck": "src/reporting/build_deck.py",
}

_t0 = time.time()


def _log(msg: str) -> None:
    print(f"[{time.time() - _t0:6.1f}s] {msg}", flush=True)


def run_step(name: str) -> int:
    script = _SLOT_ROOT / STEPS[name]
    _log(f"=== {name}: {STEPS[name]}")
    proc = subprocess.run([sys.executable, str(script)], cwd=str(_SLOT_ROOT))
    if proc.returncode == 0:
        _log(f"--- {name}: OK")
    else:
        _log(f"!!! {name}: FAILED (exit {proc.returncode})")
    return proc.returncode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", nargs="+", choices=list(STEPS),
                    help="run only these steps, in the order given")
    ap.add_argument("--skip", nargs="+", choices=list(STEPS), default=[],
                    help="run everything except these steps")
    ap.add_argument("--list", action="store_true", help="print the steps and exit")
    ap.add_argument("--keep-going", action="store_true",
                    help="continue after a failing step instead of stopping")
    args = ap.parse_args()

    if args.list:
        for n, s in STEPS.items():
            print(f"  {n:9s} {s}")
        return 0

    plan = args.steps if args.steps else [n for n in STEPS if n not in args.skip]
    _log(f"plan: {' -> '.join(plan)}")

    failed: list[str] = []
    for name in plan:
        if run_step(name) != 0:
            failed.append(name)
            if not args.keep_going:
                _log(f"stopping after {name} (use --keep-going to continue)")
                break

    if failed:
        _log(f"DONE with failures: {', '.join(failed)}")
        return 1
    _log(f"DONE — {len(plan)} step(s) OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
