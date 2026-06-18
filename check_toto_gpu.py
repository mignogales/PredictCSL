"""Quick GPU sanity check for Toto 2.0. Run on the SERVER.

    python check_toto_gpu.py
    python check_toto_gpu.py --device cuda:1 --model Datadog/Toto-2.0-313m

Tells you (1) where the params live, (2) whether a forecast moves GPU memory,
and (3) how long the forward pass takes — a CPU fallback is obvious on all three.
"""
import argparse
import time

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Datadog/Toto-2.0-313m")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--horizon", type=int, default=64)
    ap.add_argument("--length", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--patch", type=int, default=32)
    args = ap.parse_args()

    print(f"torch {torch.__version__} | cuda available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("!! CUDA not visible to torch — Toto cannot use a GPU here.")
        return
    print(f"cuda device count: {torch.cuda.device_count()}")
    dev = torch.device(args.device)
    idx = dev.index or 0
    print(f"target device: {dev}  ({torch.cuda.get_device_name(idx)})")

    from toto2 import Toto2Model

    torch.cuda.reset_peak_memory_stats(idx)
    mem_before = torch.cuda.memory_allocated(idx)

    model = Toto2Model.from_pretrained(args.model).to(dev).eval()

    # (1) Where do the weights actually live?
    devs = {str(p.device) for p in model.parameters()}
    n_gpu = sum(p.numel() for p in model.parameters() if p.is_cuda)
    n_tot = sum(p.numel() for p in model.parameters())
    print(f"\nparam devices: {devs}")
    print(f"params on GPU: {n_gpu/1e6:.1f}M / {n_tot/1e6:.1f}M "
          f"({100*n_gpu/max(n_tot,1):.0f}%)")
    mem_weights = torch.cuda.memory_allocated(idx) - mem_before
    print(f"GPU mem after load: +{mem_weights/1e6:.1f} MB")

    # (2) Build an input exactly like predict_toto does (pad to patch multiple).
    b, length = args.batch, args.length
    pad = (-length) % args.patch
    seqs = torch.randn(b, length, device=dev)
    if pad:
        seqs = torch.cat([seqs.new_zeros((b, pad)), seqs], dim=1)
    target = seqs.unsqueeze(1)
    target_mask = torch.ones_like(target, dtype=torch.bool)
    if pad:
        target_mask[:, :, :pad] = False
    series_ids = torch.arange(b, device=dev, dtype=torch.long).unsqueeze(1)

    # (3) Time the forecast + watch peak memory.
    torch.cuda.synchronize(idx)
    t0 = time.perf_counter()
    with torch.no_grad():
        q = model.forecast(
            {"target": target, "target_mask": target_mask, "series_ids": series_ids},
            horizon=args.horizon,
            has_missing_values=bool(pad),
        )
    torch.cuda.synchronize(idx)
    dt = time.perf_counter() - t0

    out_dev = q.device if torch.is_tensor(q) else "n/a"
    peak = torch.cuda.max_memory_allocated(idx)
    print(f"\nforecast output device: {out_dev}")
    print(f"forecast wall time: {dt*1000:.1f} ms  (batch={b}, len={length}, h={args.horizon})")
    print(f"peak GPU mem: {peak/1e6:.1f} MB")

    ok = (n_gpu == n_tot) and torch.is_tensor(q) and q.is_cuda and peak > mem_weights
    print("\n==> Toto IS running on the GPU." if ok
          else "\n==> WARNING: Toto does NOT appear fully on the GPU — see above.")


if __name__ == "__main__":
    main()
