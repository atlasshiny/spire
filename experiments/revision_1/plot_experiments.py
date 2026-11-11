#!/usr/bin/env python3
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "/home/manny/experiments"

# --- Experiment 2 plot ---
with open(os.path.join(OUT, "exp2_spire_vs_inline.csv")) as f:
    rows = list(csv.DictReader(f))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ds = [int(r["d_sae"]) for r in rows]
inline = [float(r["inline_tps"]) for r in rows]
spire = [float(r["spire_tps"]) for r in rows]
speedup = [float(r["speedup_vs_inline"]) for r in rows]

x = np.arange(len(ds))
w = 0.35
ax1.bar(x - w/2, inline, w, label="Inline (sync)", color="#4C72B0")
ax1.bar(x + w/2, spire, w, label="Spire (async)", color="#DD8452")
ax1.set_xticks(x)
ax1.set_xticklabels([str(d) for d in ds])
ax1.set_xlabel("Dictionary size (d_sae)")
ax1.set_ylabel("Throughput (tokens/s)")
ax1.set_title("Spire vs Inline SAE Hook")
ax1.legend()

colors = ["#2ca02c" if v > 1 else "#d62728" for v in speedup]
ax2.bar([str(d) for d in ds], speedup, color=colors, width=0.6)
ax2.axhline(1.0, color="black", linestyle="--", linewidth=1.5, label="break-even")
for i, v in enumerate(speedup):
    ax2.text(i, v + 0.02, f"{v:.2f}x", ha="center", fontsize=9, fontweight="bold")
ax2.set_xlabel("Dictionary size (d_sae)")
ax2.set_ylabel("Speedup (inline/spire)")
ax2.set_title("Spire Speedup vs Inline")
ax2.legend()

fig.tight_layout()
fig.savefig(os.path.join(OUT, "exp2_spire_vs_inline.png"), dpi=150)
plt.close(fig)
print("Saved exp2 plot")

# --- Experiment 3 plot ---
with open(os.path.join(OUT, "exp3_subset_latents.csv")) as f:
    rows = list(csv.DictReader(f))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
dims = [int(r["sae_dim"]) for r in rows]
ms = [float(r["ms_per_call"]) for r in rows]
savings = [float(r["savings_pct"]) for r in rows]

ax1.plot(dims, ms, "o-", linewidth=2.5, color="#4C72B0", markersize=8)
ax1.set_xscale("log", base=2)
ax1.set_xlabel("SAE dimension (subset size)")
ax1.set_ylabel("Encoder latency (ms)")
ax1.set_title("Encoder Cost vs Dictionary Size")
ax1.grid(True, ls="--", alpha=0.3)

ax2.bar([str(d) for d in dims], savings, color="#2ca02c", width=0.6)
for i, v in enumerate(savings):
    ax2.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold")
ax2.set_xlabel("Subset size (of 24576)")
ax2.set_ylabel("Savings (%)")
ax2.set_title("Cost Savings from Partial Computation")
ax2.set_xticklabels([str(d) for d in dims], rotation=45, ha="right")

fig.tight_layout()
fig.savefig(os.path.join(OUT, "exp3_subset_latents.png"), dpi=150)
plt.close(fig)
print("Saved exp3 plot")