"""Scaling law and Muon/MoE ablations on TinyStories."""
import argparse
import json
import os
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MaxNLocator, NullLocator, ScalarFormatter

from train import TrainConfig, train

FULL = TrainConfig(max_steps=8000, warmup_steps=200, eval_every=400)  # 65.5M tokens per run with normal corpus
# cpu smoke test, tiny because muon's bf16 newton-schulz is slow on cpu
QUICK = TrainConfig(dim=64, n_layers=2, n_heads=4, n_kv_heads=2, seq_len=64, batch_size=16, max_steps=50,
                    warmup_steps=5, eval_every=10, eval_batches=4)

# (dim, n_layers, n_heads, n_kv_heads), 0.1M to 10M params, ~3x apart
SIZES = [(64, 2, 4, 2), (96, 3, 4, 2), (128, 5, 4, 2), (256, 4, 8, 4), (384, 6, 8, 4)]
# default model, one change per run
ABLATIONS = [
    ("dense_adamw", "dense + AdamW", {}),
    ("dense_muon", "dense + Muon", {"optimizer": "muon"}),
    ("moe_adamw", "MoE (8 experts, top-2) + AdamW", {"n_experts": 8, "top_k": 2}),
]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE = "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#52514e",
    "xtick.color": "#898781", "ytick.color": "#898781", "xtick.labelcolor": "#52514e", "ytick.labelcolor": "#52514e",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
    "grid.color": "#e1e0d9", "grid.linewidth": 0.8, "legend.frameon": False,
})


def scaling(base):
    runs = []
    for dim, n_layers, n_heads, n_kv_heads in SIZES:
        r = train(replace(base, dim=dim, n_layers=n_layers, n_heads=n_heads, n_kv_heads=n_kv_heads,
                          eval_every=base.max_steps, out=f"out/scaling_{dim}x{n_layers}.pt"))
        runs.append({"dim": dim, "n_layers": n_layers, "params": r["params"], "val_loss": r["val_loss"]})
    N, L = np.array([(r["params"], r["val_loss"]) for r in runs]).T
    # pure power law, no irreducible loss term
    slope, intercept = np.polyfit(np.log(N), np.log(L), 1)
    a, alpha = np.exp(intercept), -slope
    tokens = base.max_steps * base.batch_size * base.seq_len
    with open("out/scaling.json", "w") as f:
        json.dump({"tokens_per_run": tokens, "fit": {"a": a, "alpha": alpha}, "runs": runs}, f, indent=1)

    print(f"\nL(N) = {a:.2f} * N^-{alpha:.3f}, {tokens / 1e6:.1f}M tokens per run\n")
    print("| non-embedding params | dim | layers | val loss | fit |\n|---:|---:|---:|---:|---:|")
    for r in runs:
        print(f"| {r['params'] / 1e6:.2f}M | {r['dim']} | {r['n_layers']} | {r['val_loss']:.3f} "
              f"| {a * r['params'] ** -alpha:.3f} |")

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    xs = np.geomspace(N.min() / 1.5, N.max() * 1.5)
    ax.loglog(xs, a * xs ** -alpha, "--", color="#898781", lw=1.5, label=f"fit: $L = {a:.2f}\\,N^{{-{alpha:.3f}}}$")
    ax.loglog(N, L, "o", color=COLORS[0], ms=9, mec=SURFACE, mew=2, label="measured")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda n, _: f"{n / 1e6:g}M"))
    ax.grid(True, which="minor", color="#f0efec", lw=0.6)
    # small loss range, log ticks would only give 1-2 labels
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set(xlabel="non-embedding parameters N", ylabel="validation loss",
           title=f"Scaling law on TinyStories ({tokens / 1e6:.1f}M tokens per model)")
    ax.legend()
    fig.savefig("assets/scaling.png", dpi=150, bbox_inches="tight")
    print("\nsaved out/scaling.json and assets/scaling.png")


def ablations(base):
    runs = {}
    for name, label, change in ABLATIONS:
        r = train(replace(base, out=f"out/ablation_{name}.pt", **change))
        runs[name] = {"label": label, "val_loss": r["val_loss"], "active_params": r["active_params"],
                      "total_params": r["params"], "tok_s": r["tok_s"],
                      "curve": [[x["tokens"], x["val_loss"]] for x in r["records"] if "val_loss" in x]}
    tokens = base.max_steps * base.batch_size * base.seq_len
    with open("out/ablations.json", "w") as f:
        json.dump({"tokens_per_run": tokens, "runs": runs}, f, indent=1)

    print(f"\n{tokens / 1e6:.1f}M tokens per run\n")
    print("| run | val loss | active params | total params | tok/s |\n|---|---:|---:|---:|---:|")
    for r in runs.values():
        print(f"| {r['label']} | {r['val_loss']:.3f} | {r['active_params'] / 1e6:.2f}M "
              f"| {r['total_params'] / 1e6:.2f}M | {r['tok_s']:,.0f} |")

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for r, color in zip(runs.values(), COLORS):
        t, loss = np.array(r["curve"]).T
        ax.plot(t / 1e6, loss, color=color, lw=2, label=f"{r['label']}: {r['val_loss']:.3f}")
        ax.plot(t[-1] / 1e6, loss[-1], "o", color=color, ms=8, mec=SURFACE, mew=2)
    ax.set(xlabel="training tokens (M)", ylabel="validation loss",
           title="Same tokens, same active compute per token")
    ax.legend()
    fig.savefig("assets/ablations.png", dpi=150, bbox_inches="tight")
    print("\nsaved out/ablations.json and assets/ablations.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("experiment", choices=["scaling", "ablations"])
    p.add_argument("--quick", action="store_true", help="tiny CPU smoke test (~1-2 min) instead of the full runs")
    p.add_argument("--compile", action="store_true", help="torch.compile the models (CUDA only)")
    args = p.parse_args()
    os.makedirs("assets", exist_ok=True)
    base = replace(QUICK if args.quick else FULL, compile=args.compile)
    {"scaling": scaling, "ablations": ablations}[args.experiment](base)
