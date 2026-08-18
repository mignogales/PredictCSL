"""Generate compact, real forecasting examples for every synthetic DGP.

The shaded part of each panel is exactly the input supplied to the model; the
black trace to its right is withheld while the forecast is made.  Forecasts
are cached per model, which lets models with separate environments (notably
ToTo) contribute to the same figure.
"""
from __future__ import annotations

import argparse
import gc
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import Patch

from experiments import models_config
from experiments.build_context_length_dataset import _forecast_uniform, setup_model
from experiments.synth_param_sweeps import (
    MAX_WINDOW, gen_ar, gen_break_age, gen_canonical, gen_delay, gen_memory,
    gen_missing_gap, gen_multiscale, gen_period, gen_period_drift, gen_regime,
    gen_seasonality,
)

HORIZON = 256
DEFAULT_MODELS = ["Chronos2-Small", "Moirai2-Small", "Toto-2.0-313m", "FlowState-R1"]


@dataclass(frozen=True)
class Case:
    key: str
    title: str
    generator: object
    kwargs: dict
    short: int
    enough: int


# The two panels use exactly the same normalized coordinates as the sweep:
# L/q=1 and L/q=8, where q is the DGP's controlling parameter.  Values whose
# 8q context would exceed a model cap use a smaller representative q instead.
CASES = (
    Case("period", "Periodicity ($T=256$)", gen_period, {"T": 256}, 256, 2048),
    Case("seasonality", "Seasonality ($S=256$)", gen_seasonality, {"S": 256}, 256, 2048),
    Case("ar_order", "AR order ($p=4,\\;\\tau=256$)", gen_ar, {"tau": 256, "order": 4}, 256, 2048),
    Case("memory", "Memory ($\\tau=256$)", gen_memory, {"tau": 256}, 256, 2048),
    Case("delay", "Delayed dependency ($d=512$)", gen_delay, {"d": 512, "horizon": HORIZON}, 512, 4096),
    Case("regime", "Regime duration ($D=512$)", gen_regime, {"D": 512}, 512, 4096),
    Case("horizon", "Forecast horizon ($h=256$)", gen_canonical, {}, 256, 2048),
    Case("break_age", "Change-point recency ($A=512$)", gen_break_age, {"A": 512}, 512, 4096),
    Case("snr", "Signal-to-noise ($T=256,\\;\\sigma=1.0$)", gen_period, {"T": 256, "noise_std": 1.0}, 256, 2048),
    Case("multiscale", "Multiscale ($T=64,\\;k=8$)", gen_multiscale, {"T": 64, "k": 8}, 512, 4096),
    # A short correlation time makes the local frequency evolution visible in
    # an appendix panel.  The sweep itself still spans M=256...4096.
    Case("period_drift", "Period drift ($M=256$)", gen_period_drift, {"M": 256}, 256, 2048),
    Case("missing_gap", "Missing gap ($G=512$)", gen_missing_gap, {"G": 512}, 512, 4096),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default="logs/experiments/synth_param_sweeps/summary_plots")
    parser.add_argument("--compute-only", action="store_true", help="Generate/update per-model cached forecasts only.")
    parser.add_argument("--plot-only", action="store_true", help="Render from cached forecasts; do not load a model.")
    parser.add_argument("--assemble-only", action="store_true", help="Build paper sheets from existing per-process PNGs.")
    return parser.parse_args()


def make_series() -> list[tuple[Case, np.ndarray]]:
    return [
        (case, case.generator(np.random.RandomState(20260812 + index), MAX_WINDOW + HORIZON, **case.kwargs))
        for index, case in enumerate(CASES)
    ]


def cache_path(cache_dir: Path, display: str) -> Path:
    return cache_dir / f"{display.replace('/', '_')}.npz"


def compute_forecasts(args: argparse.Namespace, series: list[tuple[Case, np.ndarray]], catalog: dict) -> None:
    cache_dir = Path(args.output_dir) / "forecast_example_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    contexts = np.asarray([(case.short, case.enough) for case, _ in series], dtype=np.int32)
    keys = np.asarray([case.key for case, _ in series])
    for display in args.models:
        model_id, family = catalog[display]
        base = setup_model(family, model_id, args.device)
        predictions = np.empty((len(series), 2, HORIZON), dtype=np.float32)
        for index, (case, values) in enumerate(series):
            for condition, context in enumerate((case.short, case.enough)):
                history = torch.from_numpy(np.ascontiguousarray(values[MAX_WINDOW - context:MAX_WINDOW])).view(1, context, 1)
                prediction = _forecast_uniform(family, base, model_id, history, context, HORIZON, batch_size=1, device=args.device)
                predictions[index, condition] = prediction[0].detach().cpu().numpy()
        np.savez_compressed(cache_path(cache_dir, display), keys=keys, contexts=contexts, predictions=predictions)
        print(f"cached {display}: {cache_path(cache_dir, display)}")
        del base
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()


def load_cached(output: Path, models: list[str]) -> dict[str, np.ndarray]:
    cache_dir = output / "forecast_example_cache"
    cached = {}
    expected_keys = [case.key for case in CASES]
    for display in models:
        path = cache_path(cache_dir, display)
        if not path.exists():
            raise SystemExit(f"Missing cached forecasts for {display}: {path}")
        data = np.load(path)
        if list(data["keys"]) != expected_keys:
            raise SystemExit(f"Cached DGP order does not match for {display}: {path}")
        cached[display] = data["predictions"]
    return cached


def assemble_paper_figures(output: Path) -> None:
    figure_dir = output / "09_forecast_examples"
    # Paper-ready sheets: three processes per figure, each retaining its full
    # vertically stacked insufficient/sufficient comparison.  Three compact
    # process panels make better use of a full portrait paper page.
    paper_dir = output / "paper_figures"
    paper_dir.mkdir(parents=True, exist_ok=True)
    processes_per_sheet = 3
    for sheet_index, start in enumerate(range(0, len(CASES), processes_per_sheet), start=1):
        sheet_cases = CASES[start:start + processes_per_sheet]
        sheet, axes = plt.subplots(len(sheet_cases), 1,
                                   figsize=(12.2, 6.25 * len(sheet_cases)))
        axes = np.atleast_1d(axes)
        for axis, case in zip(axes, sheet_cases):
            axis.imshow(plt.imread(figure_dir / f"{case.key}.png"))
            axis.axis("off")
        sheet.tight_layout(pad=0.25, h_pad=0.45)
        stem = f"synthetic_forecasts_{sheet_index:02d}_{sheet_cases[0].key}_{sheet_cases[-1].key}"
        for extension in ("png", "pdf"):
            sheet.savefig(paper_dir / f"{stem}.{extension}", dpi=220,
                         bbox_inches="tight")
        plt.close(sheet)
        print(paper_dir / f"{stem}.png")

    # A browsable appendix index.  One process per row keeps both context
    # conditions and forecast traces legible in a long-form appendix PDF.
    overview, axes = plt.subplots(len(CASES), 1, figsize=(12.2, 6.25 * len(CASES)))
    for axis, case in zip(axes.flat, CASES):
        axis.imshow(plt.imread(figure_dir / f"{case.key}.png"))
        axis.axis("off")
    overview.suptitle("Synthetic forecasting examples — all processes", fontsize=18, y=0.998)
    overview.tight_layout(rect=(0, 0, 1, 0.994), h_pad=0.55)
    for extension in ("png", "pdf"):
        overview.savefig(figure_dir / f"all_processes_overview.{extension}", dpi=180,
                         bbox_inches="tight")
    plt.close(overview)
    print(figure_dir / "all_processes_overview.png")


def plot_cases(output: Path, models: list[str], series: list[tuple[Case, np.ndarray]]) -> None:
    forecasts = load_cached(output, models)
    figure_dir = output / "09_forecast_examples"
    figure_dir.mkdir(parents=True, exist_ok=True)
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(models)))
    for index, (case, values) in enumerate(series):
        # Stack the two conditions so each receives the full page width.  This
        # is particularly important when the sufficient input is long: the
        # held-out horizon and the competing forecasts remain legible.
        fig, axes = plt.subplots(2, 1, figsize=(10.8, 5.8), sharey=True)
        for condition, (context, label, shade) in enumerate(((case.short, "Insufficient context", "#F59E0B"), (case.enough, "Sufficient context", "#10B981"))):
            ax = axes[condition]
            shown = min(max(4 * context, 768), 4096)
            time = np.arange(-shown, HORIZON)
            ax.plot(time, values[MAX_WINDOW - shown:MAX_WINDOW + HORIZON], color="#111827", lw=1.05, label="True series")
            ax.axvspan(-context, 0, color=shade, alpha=0.18, label="Model input")
            ax.axvline(0, color="#374151", ls="--", lw=0.9, label="Forecast origin")
            for model_index, display in enumerate(models):
                ax.plot(np.arange(HORIZON), forecasts[display][index, condition], color=colors[model_index], lw=1.25, label=display)
            ax.set_title(f"{label} ($L/q={context / case.short:g}$; $L={context}$)", fontsize=10)
            ax.set_xlabel("steps relative to forecast origin")
            ax.grid(alpha=0.2)
        for ax in axes:
            ax.set_ylabel("standardized value")
        handles, labels = axes[0].get_legend_handles_labels()
        input_index = labels.index("Model input")
        handles[input_index] = (Patch(facecolor="#F59E0B", edgecolor="#F59E0B", alpha=0.18), Patch(facecolor="#10B981", edgecolor="#10B981", alpha=0.18))
        labels[input_index] = "Model input (amber / teal)"
        fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False, fontsize=8.5, handler_map={tuple: HandlerTuple(ndivide=None)})
        fig.suptitle(case.title, fontsize=12)
        fig.tight_layout(rect=(0, 0.09, 1, 0.94))
        for extension in ("png", "pdf"):
            fig.savefig(figure_dir / f"{case.key}.{extension}", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(figure_dir / f"{case.key}.png")
    assemble_paper_figures(output)


def main() -> None:
    args = parse_args()
    if sum((args.compute_only, args.plot_only, args.assemble_only)) > 1:
        raise SystemExit("choose at most one of --compute-only, --plot-only, and --assemble-only")
    catalog = {display: (model_id, family) for model_id, family, display in models_config.catalog()}
    unknown = [name for name in args.models if name not in catalog]
    if unknown:
        raise SystemExit(f"Unknown models: {unknown}")
    series = make_series()
    output = Path(args.output_dir)
    if args.assemble_only:
        assemble_paper_figures(output)
        return
    if not args.plot_only:
        compute_forecasts(args, series, catalog)
    if not args.compute_only:
        plot_cases(output, args.models, series)


if __name__ == "__main__":
    main()
