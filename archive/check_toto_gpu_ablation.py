"""GPU sanity check for Toto using the ACTUAL ablation wrappers. Run on SERVER.

    python check_toto_gpu_ablation.py
    python check_toto_gpu_ablation.py --device cuda --length 2048 --horizon 64

Imports load_toto / predict_toto straight from test_window_ablation_gifteval_v5
so it exercises the exact code the ablation runs, then reports where the params
live, the output device, peak GPU memory, and wall time.
"""
import argparse
import time

import torch

from experiments.test_window_ablation_gifteval_v5 import (
    load_toto, predict_toto, MODELS,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="defaults to the toto entry in MODELS")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--horizon", type=int, default=64)
    ap.add_argument("--length", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    model_id = args.model or next(m[0] for m in MODELS if m[1] == "toto")
    print(f"torch {torch.__version__} | cuda available: {torch.cuda.is_available()}")
    if args.device == "cuda" and not torch.cuda.is_available():
        print("!! CUDA not visible to torch — ablation will fall back to CPU.")
        return
    idx = torch.cuda.current_device() if args.device == "cuda" else 0
    if args.device == "cuda":
        print(f"device: cuda:{idx}  ({torch.cuda.get_device_name(idx)})")
        torch.cuda.reset_peak_memory_stats(idx)
        mem0 = torch.cuda.memory_allocated(idx)

    # ---- Load via the ablation's own loader -------------------------------
    model = load_toto(model_id, args.device)
    devs = {str(p.device) for p in model.parameters()}
    n_gpu = sum(p.numel() for p in model.parameters() if p.is_cuda)
    n_tot = sum(p.numel() for p in model.parameters())
    print(f"\nmodel: {model_id}")
    print(f"param devices: {devs}")
    print(f"params on GPU: {n_gpu/1e6:.1f}M / {n_tot/1e6:.1f}M "
          f"({100*n_gpu/max(n_tot,1):.0f}%)")

    # ---- Build a batch shaped like the ablation's dataloader --------------
    # predict_toto expects batches of dicts with x:(B,L,1), y:(B,H,1) tensors.
    x = torch.randn(args.batch, args.length, 1)
    y = torch.randn(args.batch, args.horizon, 1)
    batches = [{"x": x, "y": y}]

    if args.device == "cuda":
        torch.cuda.synchronize(idx)
    t0 = time.perf_counter()
    fr, tgts = predict_toto(model, batches, args.horizon, args.device)
    if args.device == "cuda":
        torch.cuda.synchronize(idx)
    dt = time.perf_counter() - t0

    print(f"\nforecast median device: {fr.median.device}")
    print(f"wall time: {dt*1000:.1f} ms  (batch={args.batch}, len={args.length}, h={args.horizon})")
    if args.device == "cuda":
        peak = torch.cuda.max_memory_allocated(idx)
        print(f"peak GPU mem: {(peak-mem0)/1e6:.1f} MB")
        ok = n_gpu == n_tot and fr.median.is_cuda and peak > mem0
        print("\n==> Toto IS running on the GPU in the ablation path." if ok
              else "\n==> WARNING: Toto is NOT fully on the GPU — see above.")


if __name__ == "__main__":
    main()
