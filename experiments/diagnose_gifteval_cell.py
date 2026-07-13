"""
Pinpoint why a GiftEval (ge_name, term) cell is rejected at load — the systemic
``no_stage3_cell`` case from ``analyze_leftover_cells`` (e.g. CarParts, Solar-W).

Stage 3 wraps ``GiftEvalDataset`` + ``GiftEvalCache`` construction in a try/except
and, on failure, prints ``SKIP dataset … (exc)`` and writes NO cell — so the cell
never reaches stage 4 and the bar's ``n`` is short. This script reproduces that
construction OUT of the run loop and reports the actual cause, trying both
``to_univariate`` values so the fix (a config edit in ``datasets_config``) is
obvious:

  * constructor raises          -> unsupported (name, term); the traceback says so
  * target.ndim > 1             -> needs ``to_univariate=True`` in datasets_config
  * every label shorter than H  -> "No valid test instances" (series too short for
                                    this term's horizon) — drop the cell or pick a
                                    term whose horizon fits
  * builds fine with a flag     -> datasets_config has the wrong ``to_univariate``

Run on the SERVER (needs the GIFT_EVAL data + env). Examples:

    python -m experiments.diagnose_gifteval_cell --name car_parts_with_missing --term short
    python -m experiments.diagnose_gifteval_cell --name solar/W --term short
    # sweep every cell analyze_leftover_cells flagged as no_stage3_cell:
    python -m experiments.diagnose_gifteval_cell --from-config-missing
"""

from __future__ import annotations

import argparse
import traceback
from typing import List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv

# gift_eval resolves its data dir from a storage env var (GIFT_EVAL / …), read at
# Dataset construction. Stage 3 loads it from .env at import; mirror that here so
# the probe doesn't die with "expected str … not NoneType" before it even sees
# the dataset.
load_dotenv()

from gift_eval.data import Dataset as GiftEvalDataset
from experiments import datasets_config

try:
    from colorama import Fore, init as _cinit
    _cinit()
except Exception:
    class _F:
        def __getattr__(self, _):
            return ""
    Fore = _F()                                     # type: ignore


def _probe(name: str, term: str, to_univariate: bool) -> None:
    """Construct one (name, term, to_univariate) and report what happens."""
    tag = f"{name}  term={term}  to_univariate={to_univariate}"
    try:
        ds = GiftEvalDataset(name=name, term=term, to_univariate=to_univariate)
    except Exception as exc:                         # constructor rejected it
        print(Fore.RED + f"  [{tag}] constructor FAILED: {type(exc).__name__}: {exc}"
              + Fore.RESET)
        traceback.print_exc()
        return

    freq = getattr(ds, "freq", "?")
    horizon = getattr(ds, "prediction_length", None)
    print(Fore.CYAN + f"  [{tag}] built: freq={freq}  horizon={horizon}" + Fore.RESET)

    n_total = 0
    n_multivar = 0
    n_short_label = 0
    n_valid = 0
    label_lens: List[int] = []
    ctx_lens: List[int] = []
    try:
        for test_input, test_label in ds.test_data:
            n_total += 1
            target = np.asarray(test_input["target"])
            label = np.asarray(test_label["target"])
            if target.ndim > 1 or label.ndim > 1:
                n_multivar += 1
                continue
            label_lens.append(len(label))
            ctx_lens.append(len(target))
            if horizon is not None and len(label) < horizon:
                n_short_label += 1
                continue
            n_valid += 1
    except Exception as exc:
        print(Fore.RED + f"    iterating test_data FAILED: {type(exc).__name__}: {exc}"
              + Fore.RESET)
        traceback.print_exc()
        return

    print(f"    instances: total={n_total}  multivariate(ndim>1)={n_multivar}  "
          f"label<horizon={n_short_label}  VALID={n_valid}")
    if ctx_lens:
        print(f"    context len  min/median/max = "
              f"{min(ctx_lens)}/{int(np.median(ctx_lens))}/{max(ctx_lens)}")
    if label_lens:
        print(f"    label   len  min/median/max = "
              f"{min(label_lens)}/{int(np.median(label_lens))}/{max(label_lens)}")

    # Verdict mirroring GiftEvalCache's raise conditions.
    if n_valid == 0:
        if n_multivar == n_total and n_total > 0:
            print(Fore.YELLOW + "    => VERDICT: multivariate target -> set "
                  "to_univariate=True in datasets_config." + Fore.RESET)
        elif n_short_label == (n_total - n_multivar) and n_total > 0:
            print(Fore.YELLOW + "    => VERDICT: every label shorter than the "
                  "horizon -> 'No valid test instances'. This term's horizon "
                  "doesn't fit; drop the cell or use a shorter-horizon term."
                  + Fore.RESET)
        else:
            print(Fore.YELLOW + "    => VERDICT: no valid instances (mixed cause; "
                  "see counts above)." + Fore.RESET)
    else:
        print(Fore.GREEN + f"    => VERDICT: OK — {n_valid} valid instances build "
              "cleanly with this flag. If datasets_config uses a different "
              "to_univariate, that's the bug." + Fore.RESET)


def _missing_cells() -> List[Tuple[str, str, bool]]:
    """(ge_name, term, to_univariate) for every run=True catalog cell — the
    caller can diff against a run, but here we just probe the two known offenders
    plus anything the user names. Returns the full run set in table order."""
    out = []
    for d in datasets_config.CATALOG:
        if d.run:
            out.append((d.ge_name, d.term, d.to_univariate))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", help="gift_eval config name, e.g. solar/W")
    ap.add_argument("--term", default="short", help="short | medium | long")
    ap.add_argument("--both-flags", action="store_true", default=True,
                    help="probe both to_univariate values (default on)")
    ap.add_argument("--only-flag", type=lambda s: s.lower() in ("1", "true", "yes"),
                    default=None, help="probe just this to_univariate value")
    ap.add_argument("--from-config-missing", action="store_true",
                    help="probe the two known systemic offenders "
                         "(car_parts_with_missing/short, solar/W/short)")
    args = ap.parse_args()

    probes: List[Tuple[str, str, Optional[bool]]] = []
    if args.from_config_missing:
        # The cells analyze_leftover_cells flags as no_stage3_cell for all models.
        probes += [("car_parts_with_missing", "short", None),
                   ("solar/W", "short", None)]
    if args.name:
        probes.append((args.name, args.term, args.only_flag))
    if not probes:
        ap.error("give --name or --from-config-missing")

    # Pull the datasets_config to_univariate default for context.
    cfg = {(d.ge_name, d.term): d.to_univariate for d in datasets_config.CATALOG}

    for name, term, flag in probes:
        cfg_flag = cfg.get((name, term))
        print(Fore.CYAN + f"\n=== {name}  term={term}  "
              f"(datasets_config to_univariate={cfg_flag}) ===" + Fore.RESET)
        flags = [flag] if flag is not None else [False, True]
        for f in flags:
            _probe(name, term, f)


if __name__ == "__main__":
    main()
