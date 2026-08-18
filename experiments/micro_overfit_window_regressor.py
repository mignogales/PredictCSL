#!/usr/bin/env python3
"""Strict memorization test for the continuous instance-window predictor."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.predict_context_length import MambaContextLength

WINDOWS = torch.tensor((32, 48, 64, 96, 128, 192, 256, 384, 512, 768,
                        1024, 1536, 2048, 2560, 3072, 4096, 6144, 8192))


class FlatMLP(nn.Module):
    def __init__(self, context_length: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(context_length, 512), nn.GELU(),
            nn.Linear(512, 256), nn.GELU(), nn.Linear(256, 1),
        )

    def forward(self, x, horizon_idx):
        del horizon_idx
        return self.net(x.squeeze(-1))


class MambaRegressor(nn.Module):
    def __init__(self, context_length, d_model, layers, d_state):
        super().__init__()
        self.encoder = MambaContextLength(
            context_length, 128, d_model, layers, d_state, 4, 2,
            0.0, 0.0, 1, 3,
        )

    def forward(self, x, horizon_idx):
        return self.encoder(x, horizon_idx)[0]


@torch.no_grad()
def metrics(model, x, h, target):
    model.eval()
    pred = 5.0 + 8.0 * model(x, h).squeeze(1)
    error = (pred - target).abs()
    grid = torch.log2(WINDOWS.to(device=pred.device, dtype=pred.dtype))
    predicted_index = (pred[:, None] - grid[None, :]).abs().argmin(1)
    target_index = (target[:, None] - grid[None, :]).abs().argmin(1)
    return {
        "mae_log2": float(error.mean()),
        "median_ae_log2": float(error.median()),
        "max_ae_log2": float(error.max()),
        "nearest_grid_accuracy": float((predicted_index == target_index).float().mean()),
    }


def train_one(name, model, x, h, target, steps, lr):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    y = ((target - 5.0) / 8.0).unsqueeze(1)
    history = []
    for step in range(1, steps + 1):
        pred = model(x, h)
        loss = torch.nn.functional.mse_loss(pred, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        if step == 1 or step % 100 == 0 or step == steps:
            row = {"step": step, "mse_normalized": float(loss), **metrics(model, x, h, target)}
            history.append(row)
            print(name, row, flush=True)
            model.train()
    return {
        "parameters": sum(p.numel() for p in model.parameters()),
        "final": metrics(model, x, h, target),
        "history": history,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--cell", default="JenaWeather-10T/long")
    p.add_argument("--device", default="cuda:3")
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    root = Path(a.data_dir)
    cells = np.load(root / "cell.npy")
    ix = np.flatnonzero(cells == a.cell)
    if not len(ix):
        raise ValueError(f"No rows found for cell {a.cell!r}")
    x = torch.from_numpy(np.ascontiguousarray(np.load(root / "contexts.npy")[ix])).float().unsqueeze(-1).to(a.device)
    h = torch.from_numpy(np.load(root / "horizon_idx.npy")[ix]).long().to(a.device)
    target = torch.from_numpy(np.load(root / "target_log2.npy")[ix]).float().to(a.device)
    specs = {
        "flat_mlp": FlatMLP(x.shape[1]),
        "mamba_current": MambaRegressor(x.shape[1], 128, 2, 16),
        "mamba_large": MambaRegressor(x.shape[1], 256, 4, 32),
    }
    report = {"cell": a.cell, "n": len(ix), "unique_targets": int(target.unique().numel()), "models": {}}
    for name, model in specs.items():
        model = model.to(a.device)
        report["models"][name] = train_one(name, model, x, h, target, a.steps, 1e-3)
        del model
        torch.cuda.empty_cache()
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v["final"] for k, v in report["models"].items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
