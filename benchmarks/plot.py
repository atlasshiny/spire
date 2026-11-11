#!/usr/bin/env python3
"""Plotting pipeline for every Spire Google Benchmark executable.

Usage:
    python benchmarks/plot.py [results_dir] [output_dir]

`results_dir` should contain Google Benchmark CSV files produced by
`benchmarks/bench_and_plot.sh`, named after the executable that produced them
(e.g. `sae_gemm_benchmark.csv`).  The script dispatches on the file stem so each
benchmark gets plots tailored to the metrics it actually reports.

All figures are written under `output_dir` (default: ./plots), one subdirectory
per benchmark, and a summary of what was produced is printed at the end.
"""

import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

UNIT_TO_MS = {"ns": 1e-6, "us": 1e-3, "ms": 1.0, "s": 1e3}

NUMERIC_COLUMNS = [
    "iterations",
    "real_time",
    "cpu_time",
    "bytes_per_second",
    "items_per_second",
]

PALETTE = {
    "Kernel": "#4C72B0",
    "GEMM": "#DD8452",
    "kernel": "#4C72B0",
    "gemm": "#DD8452",
}


def _detect_gpu_label():
    env = os.environ.get("GPU_LABEL")
    if env:
        return env
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL, text=True, timeout=5,
        ).strip().splitlines()
        if out and out[0]:
            return out[0].strip()
    except (FileNotFoundError, subprocess.SubprocessError, Exception):
        pass
    return "NVIDIA GPU"

GPU_LABEL = _detect_gpu_label()


def _find_header_row(path):
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for index, line in enumerate(handle):
            if line.lstrip().startswith("name,"):
                return index
    return 0

def load_csv(path):
    header_row = _find_header_row(path)
    df = pd.read_csv(path, skiprows=header_row)

    if df.empty or "name" not in df.columns:
        return pd.DataFrame()

    df["source"] = os.path.splitext(os.path.basename(path))[0]

    mean_mask = df["name"].str.endswith("/mean")
    stddev_mask = df["name"].str.endswith("/stddev")

    if mean_mask.any():
        mean_df = df[mean_mask].copy()
        stddev_df = df[stddev_mask].copy()

        mean_df["name"] = mean_df["name"].str.replace("/mean", "", regex=False)
        stddev_df["name"] = stddev_df["name"].str.replace("/stddev", "", regex=False)

        if not stddev_df.empty:
            stddev_subset = stddev_df[["name", "real_time", "time_unit"]].rename(columns={"real_time": "real_time_stddev"})
            mean_df = mean_df.merge(stddev_subset, on=["name", "time_unit"], how="left")

        df = mean_df

    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    unit = df["time_unit"] if "time_unit" in df.columns else pd.Series(["ns"] * len(df))
    df["latency_ms"] = df["real_time"] * unit.map(UNIT_TO_MS).fillna(1e-6)
    df["latency_ms"] = df["latency_ms"].fillna(df["real_time"] * 1e-6)

    if "real_time_stddev" in df.columns:
        df["latency_stddev_ms"] = df["real_time_stddev"] * unit.map(UNIT_TO_MS).fillna(1e-6)
    else:
        df["latency_stddev_ms"] = 0.0

    if "label" not in df.columns:
        df["label"] = ""
    df["label"] = df["label"].fillna("")

    return df

def load_all(results_dir):
    frames = []
    if not os.path.isdir(results_dir):
        print(f"Results directory not found: {results_dir}")
        return pd.DataFrame()

    for entry in sorted(os.listdir(results_dir)):
        if not entry.endswith(".csv"):
            continue
        path = os.path.join(results_dir, entry)
        frame = load_csv(path)
        if frame.empty:
            print(f"  skipping empty/unknown CSV: {entry}")
            continue
        print(f"  loaded {entry} ({len(frame)} rows)")
        frames.append(frame)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

def extract_batch(name):
    match = re.search(r"_b(\d+)", name)
    return int(match.group(1)) if match else np.nan

def extract_shape(name):
    match = re.search(r"b(\d+)_h(\d+)_s(\d+)", name)
    if match:
        return tuple(int(group) for group in match.groups())
    return (np.nan, np.nan, np.nan)

def extract_int_arg(name):
    match = re.search(r"/(\d+)(?:/|$)", name)
    return int(match.group(1)) if match else np.nan

def human_bytes(value):
    if pd.isna(value):
        return "?"
    value = float(value)
    for unit in ["B", "KiB", "MiB", "GiB"]:
        if value < 1024.0 or unit == "GiB":
            return f"{value:.0f}{unit}" if unit == "B" else f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{value:.1f}GiB"

def save(fig, output_dir, name, produced):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, name)
    sns.despine()
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    produced.append(path)

def truncate_label(label, max_len=32):
    if not isinstance(label, str) or len(label) <= max_len:
        return label
    return label[:max_len - 3] + "..."

def safe(value, default=0.0):
    """Coerce possibly-missing counter values to a float."""
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default

def annotate_bars(ax, fmt="{:.2f}", fontsize=8):
    for container in ax.containers:
        ax.bar_label(container, fmt=fmt, fontsize=fontsize, padding=2)

def style_axes(ax, title, xlabel, ylabel, better=None):
    suffix = ""
    if better == "higher":
        suffix = "  (higher is better)"
    elif better == "lower":
        suffix = "  (lower is better)"
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel + suffix, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, which="major", ls="--", alpha=0.35)
    ax.tick_params(labelsize=10)

def style_figure(fig, subtitle=None):
    if subtitle:
        fig.suptitle(subtitle, fontsize=10, color="gray", y=1.02)

def plot_sae_gemm(df, output_dir, produced):
    for task in ["Encoder", "Decoder"]:
        subset = df[df["name"].str.startswith(f"BM_{task}")].copy()
        if subset.empty:
            continue

        subset["backend"] = np.where(
            subset["name"].str.contains("gemm", case=False), "GEMM", "Kernel"
        )
        subset = subset[subset["name"].str.contains("gemm|kernel", case=False)]
        subset["batch"] = subset["name"].apply(extract_batch)
        subset = subset.dropna(subset=["batch"])
        if subset.empty:
            continue

        agg = (
            subset.groupby(["batch", "backend"], as_index=False).agg({"latency_ms": "mean", "latency_stddev_ms": "mean"})
        )

        fig, ax = plt.subplots(figsize=(9, 5.5))
        sns.lineplot(
            data=agg, x="batch", y="latency_ms", hue="backend",
            marker="o", linewidth=2.5, ax=ax,
            palette={"Kernel": PALETTE["Kernel"], "GEMM": PALETTE["GEMM"]},
        )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.errorbar(agg["batch"], agg["latency_ms"], yerr=agg["latency_stddev_ms"], fmt='none', capsize=3, color='black', alpha=0.6, elinewidth=1)
        style_axes(ax, f"SAE {task}: latency vs batch", "Batch size", "Latency (ms)", better="lower")
        ax.grid(True, which="both", ls="--", alpha=0.3)
        style_figure(fig, GPU_LABEL)
        save(fig, output_dir, f"{task.lower()}_latency.png", produced)

        pivot = agg.pivot(index="batch", columns="backend", values="latency_ms")
        if {"Kernel", "GEMM"}.issubset(pivot.columns):
            ratio = (pivot["Kernel"] / pivot["GEMM"]).dropna()
            fig, ax = plt.subplots(figsize=(9, 5.5))
            colors = ["#2ca02c" if v > 1 else "#d62728" for v in ratio.values]
            ax.bar(ratio.index.astype(str), ratio.values, color=colors, width=0.6)
            ax.axhline(1.0, color="black", linestyle="--", linewidth=1.5, label="break-even")
            for i, (idx, v) in enumerate(ratio.items()):
                ax.text(i, v + 0.02, f"{v:.2f}x", ha="center", fontsize=9, fontweight="bold")
            style_axes(ax, f"SAE {task}: GEMM speedup (kernel / GEMM)", "Batch size", "Speedup ratio", better="higher")
            ax.set_xticks(range(len(ratio)))
            ax.set_xticklabels(ratio.index.astype(str))
            ax.legend()
            style_figure(fig, GPU_LABEL)
            save(fig, output_dir, f"{task.lower()}_speedup.png", produced)

def _plot_single_sae_benchmark(df, output_dir, produced, title_prefix):
    data = df.copy()
    shapes = data["name"].apply(extract_shape)
    data["batch"] = [s[0] for s in shapes]
    data["hidden"] = [s[1] for s in shapes]
    data["sae"] = [s[2] for s in shapes]
    data = data.dropna(subset=["batch", "hidden", "sae"])
    if data.empty:
        return

    fixed = data[(data["hidden"] == 4096) & (data["sae"] == 16384)]
    if not fixed.empty:
        agg = fixed.groupby("batch", as_index=False).agg({"latency_ms": "mean", "latency_stddev_ms": "mean"})
        fig, ax = plt.subplots(figsize=(9, 5.5))
        sns.lineplot(data=agg, x="batch", y="latency_ms", marker="o", linewidth=2.5, color=PALETTE["Kernel"], ax=ax)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.errorbar(agg["batch"], agg["latency_ms"], yerr=agg["latency_stddev_ms"], fmt='none', capsize=3, color='black', alpha=0.6, elinewidth=1)
        style_axes(ax, f"{title_prefix}: latency vs batch (h=4096, s=16384)", "Batch size", "Latency (ms)", better="lower")
        ax.grid(True, which="both", ls="--", alpha=0.3)
        style_figure(fig, GPU_LABEL)
        save(fig, output_dir, "latency_vs_batch.png", produced)

    by_sae = data[(data["batch"] == 4) & (data["hidden"] == 4096)]
    if not by_sae.empty and by_sae["sae"].nunique() > 1:
        agg = by_sae.groupby("sae", as_index=False).agg({"latency_ms": "mean", "latency_stddev_ms": "mean"})
        fig, ax = plt.subplots(figsize=(9, 5.5))
        sns.lineplot(data=agg, x="sae", y="latency_ms", marker="s", linewidth=2.5, color="#DD8452", ax=ax)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.errorbar(agg["sae"], agg["latency_ms"], yerr=agg["latency_stddev_ms"], fmt='none', capsize=3, color='black', alpha=0.6, elinewidth=1)
        style_axes(ax, f"{title_prefix}: latency vs SAE width (batch=4, h=4096)", "SAE dimension", "Latency (ms)", better="lower")
        ax.grid(True, which="both", ls="--", alpha=0.3)
        style_figure(fig, GPU_LABEL)
        save(fig, output_dir, "latency_vs_sae.png", produced)

    by_hidden = data[(data["batch"] == 4) & (data["sae"] == 16384)]
    if not by_hidden.empty and by_hidden["hidden"].nunique() > 1:
        agg = by_hidden.groupby("hidden", as_index=False).agg({"latency_ms": "mean", "latency_stddev_ms": "mean"})
        fig, ax = plt.subplots(figsize=(9, 5.5))
        sns.lineplot(data=agg, x="hidden", y="latency_ms", marker="D", linewidth=2.5, color="#55A868", ax=ax)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.errorbar(agg["hidden"], agg["latency_ms"], yerr=agg["latency_stddev_ms"], fmt='none', capsize=3, color='black', alpha=0.6, elinewidth=1)
        style_axes(ax, f"{title_prefix}: latency vs hidden size (batch=4, s=16384)", "Hidden dimension", "Latency (ms)", better="lower")
        ax.grid(True, which="both", ls="--", alpha=0.3)
        style_figure(fig, GPU_LABEL)
        save(fig, output_dir, "latency_vs_hidden.png", produced)

    if "TFLOP/s" in data.columns:
        tp = data[["label", "TFLOP/s"]].copy()
        tp["TFLOP/s"] = tp["TFLOP/s"].apply(safe)
        tp = tp[tp["TFLOP/s"] > 0]
        if not tp.empty:
            tp = tp.sort_values("TFLOP/s", ascending=True)
            fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(tp) + 1)))
            bars = ax.barh(tp["label"], tp["TFLOP/s"], color=PALETTE["Kernel"], height=0.65)
            for bar in bars:
                ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                        f"{bar.get_width():.2f}", va="center", fontsize=8)
            style_axes(ax, f"{title_prefix}: throughput (TFLOP/s)", "TFLOP/s", "", better="higher")
            ax.set_xlim(0, tp["TFLOP/s"].max() * 1.15)
            style_figure(fig, GPU_LABEL)
            save(fig, output_dir, "throughput_tflops.png", produced)

def plot_pipeline(df, output_dir, produced):
    data = df[df["name"].str.startswith("BM_Pipeline")].copy()
    if data.empty:
        return

    data["label"] = data["label"].apply(truncate_label)

    counters = [c for c in ["p50_us", "p95_us", "p99_us"] if c in data.columns]
    if counters:
        tidy = data.melt(
            id_vars=["label"], value_vars=counters,
            var_name="percentile", value_name="latency_us",
        )
        tidy["latency_us"] = pd.to_numeric(tidy["latency_us"], errors="coerce")
        tidy = tidy.dropna(subset=["latency_us"])
        if not tidy.empty:
            fig, ax = plt.subplots(figsize=(11, max(5, 0.6 * data["label"].nunique() + 3)))
            sns.scatterplot(data=tidy, y="label", x="latency_us", hue="percentile",
                            style="percentile", s=90, ax=ax, alpha=0.85)
            if (tidy["latency_us"] > 0).all():
                ax.set_xscale("log")
            style_axes(ax, "Pipeline: end-to-end latency percentiles", "Latency (us, log scale)", "", better="lower")
            ax.legend(title="", frameon=True, framealpha=0.9)
            style_figure(fig, GPU_LABEL)
            save(fig, output_dir, "latency_percentiles.png", produced)

    if "msg/s" in data.columns:
        speed = data[["label", "msg/s"]].copy()
        speed["msg/s"] = pd.to_numeric(speed["msg/s"], errors="coerce")
        speed = speed.dropna(subset=["msg/s"])
        if not speed.empty:
            speed = speed.sort_values("msg/s", ascending=True)
            fig, ax = plt.subplots(figsize=(11, max(4, 0.6 * len(speed) + 2)))
            bars = ax.barh(speed["label"], speed["msg/s"], color="#9467bd", height=0.65)
            for bar in bars:
                ax.text(bar.get_width() + speed["msg/s"].max() * 0.01,
                        bar.get_y() + bar.get_height() / 2,
                        f"{bar.get_width():,.0f}", va="center", fontsize=8)
            style_axes(ax, "Pipeline: message throughput", "Messages / second", "", better="higher")
            ax.set_xlim(0, speed["msg/s"].max() * 1.15)
            style_figure(fig, GPU_LABEL)
            save(fig, output_dir, "throughput_msgs.png", produced)

    if "batch ms" in data.columns:
        dur = data[["label", "batch ms"]].copy()
        dur["batch ms"] = pd.to_numeric(dur["batch ms"], errors="coerce")
        dur = dur.dropna(subset=["batch ms"])
        if not dur.empty:
            dur = dur.sort_values("batch ms", ascending=True)
            fig, ax = plt.subplots(figsize=(11, max(4, 0.6 * len(dur) + 2)))
            bars = ax.barh(dur["label"], dur["batch ms"], color="#17becf", height=0.65)
            for bar in bars:
                ax.text(bar.get_width() + dur["batch ms"].max() * 0.01,
                        bar.get_y() + bar.get_height() / 2,
                        f"{bar.get_width():.2f}", va="center", fontsize=8)
            style_axes(ax, "Pipeline: batched wall-clock time", "Batch time (ms)", "", better="lower")
            ax.set_xlim(0, dur["batch ms"].max() * 1.15)
            style_figure(fig, GPU_LABEL)
            save(fig, output_dir, "batch_time.png", produced)

def plot_activation_queue(df, output_dir, produced):
    data = df[df["name"].str.startswith("BM_")].copy()
    if data.empty:
        return

    capacity = data[data["name"].str.startswith("BM_ActivationQueueCapacity")].copy()
    if not capacity.empty:
        capacity["capacity"] = capacity["name"].apply(extract_int_arg)
        capacity = capacity.dropna(subset=["capacity"]).sort_values("capacity")
        if not capacity.empty:
            fig, ax = plt.subplots(figsize=(9, 5.5))
            sns.lineplot(
                data=capacity, x="capacity", y="latency_ms",
                marker="o", linewidth=2.5, color="#2ca02c", ax=ax,
            )
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            style_axes(ax, "Activation queue: latency vs queue capacity", "Queue capacity", "Latency (ms)", better="lower")
            ax.grid(True, which="both", ls="--", alpha=0.3)
            style_figure(fig, GPU_LABEL)
            save(fig, output_dir, "capacity_scaling.png", produced)

    variants = data[~data["name"].str.startswith("BM_ActivationQueueCapacity")].copy()
    if variants.empty:
        return

    labels = variants.apply(
        lambda row: row["label"] or row["name"].split("/")[0].replace("BM_", ""),
        axis=1,
    )
    variants["variant"] = labels

    if "items_per_second" in variants.columns and variants["items_per_second"].notna().any():
        throughput = variants.dropna(subset=["items_per_second"]).sort_values("items_per_second", ascending=True)
        fig, ax = plt.subplots(figsize=(11, max(4, 0.55 * len(throughput) + 2)))
        bars = ax.barh(throughput["variant"], throughput["items_per_second"], color="#2ca02c", height=0.65)
        for bar in bars:
            ax.text(bar.get_width() + throughput["items_per_second"].max() * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{bar.get_width():,.0f}", va="center", fontsize=8)
        style_axes(ax, "Activation queue: items per second", "Items / second", "", better="higher")
        ax.set_xlim(0, throughput["items_per_second"].max() * 1.15)
        style_figure(fig, GPU_LABEL)
        save(fig, output_dir, "throughput.png", produced)

    latency = variants.dropna(subset=["latency_ms"])
    if not latency.empty:
        latency = latency.sort_values("latency_ms", ascending=True)
        fig, ax = plt.subplots(figsize=(11, max(4, 0.55 * len(latency) + 2)))
        bars = ax.barh(latency["variant"], latency["latency_ms"], color="#4C72B0", height=0.65, xerr=latency["latency_stddev_ms"])
        for bar in bars:
            ax.text(bar.get_width() + latency["latency_ms"].max() * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{bar.get_width():.4f}", va="center", fontsize=8)
        ax.set_xscale("log")
        style_axes(ax, "Activation queue: latency by operation", "Latency (ms, log scale)", "", better="lower")
        ax.set_xlim(latency["latency_ms"].min() * 0.5, latency["latency_ms"].max() * 2)
        style_figure(fig, GPU_LABEL)
        save(fig, output_dir, "latency.png", produced)

def plot_allocation(df, output_dir, produced):
    data = df[df["name"].str.startswith("BM_")].copy()
    if data.empty:
        return

    data["allocator"] = data["name"].str.split("/").str[0].str.replace("BM_", "", regex=False)
    data["allocator"] = data["label"].where(data["label"] != "", data["allocator"])
    data["bytes"] = data["name"].apply(extract_int_arg)
    data = data.dropna(subset=["bytes"])
    if data.empty:
        return

    data["size"] = data["bytes"].apply(human_bytes)
    data = data.sort_values("bytes")

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.scatterplot(data=data, x="size", y="latency_ms", hue="allocator",
                    style="allocator", s=120, ax=ax, alpha=0.9, edgecolor="white", linewidth=0.5)
    if (data["latency_ms"] > 0).all():
        ax.set_yscale("log")
    ax.errorbar(range(len(data)), data["latency_ms"], yerr=data["latency_stddev_ms"], fmt='none', capsize=3, color='gray', alpha=0.5, elinewidth=1)
    style_axes(ax, "GPU allocation: latency by size", "Allocation size", "Latency (ms, log scale)", better="lower")
    ax.legend(title="", frameon=True, framealpha=0.9, fontsize=9)
    style_figure(fig, GPU_LABEL)
    save(fig, output_dir, "latency_by_size.png", produced)

    if "bytes_per_second" in data.columns and data["bytes_per_second"].notna().any():
        bw = data.dropna(subset=["bytes_per_second"]).copy()
        bw["GB/s"] = bw["bytes_per_second"] / 1e9
        fig, ax = plt.subplots(figsize=(11, 6))
        sns.barplot(data=bw, x="size", y="GB/s", hue="allocator", ax=ax, width=0.7)
        style_axes(ax, "GPU allocation: effective bandwidth", "Allocation size", "GB/s", better="higher")
        ax.legend(title="", frameon=True, framealpha=0.9, fontsize=9)
        annotate_bars(ax, fmt="{:.1f}")
        style_figure(fig, GPU_LABEL)
        save(fig, output_dir, "bandwidth_by_size.png", produced)

PLOTTERS = {
    "sae_gemm_benchmark": lambda frame, out, done: plot_sae_gemm(frame, out, done),
    "sae_kernel_benchmark": lambda frame, out, done: _plot_single_sae_benchmark(
        frame, out, done, "SAE encoder kernel"
    ),
    "sae_decoder_benchmark": lambda frame, out, done: _plot_single_sae_benchmark(
        frame, out, done, "SAE decoder kernel"
    ),
    "pipeline_benchmark": lambda frame, out, done: plot_pipeline(frame, out, done),
    "activation_queue_benchmark": lambda frame, out, done: plot_activation_queue(frame, out, done),
    "allocation_benchmark": lambda frame, out, done: plot_allocation(frame, out, done),
}

def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "benchmark_results"
    output_root = sys.argv[2] if len(sys.argv) > 2 else "plots"

    print(f"Loading benchmark CSVs from: {results_dir}")
    all_df = load_all(results_dir)
    if all_df.empty:
        print("No benchmark data found. Run benchmarks/bench_and_plot.sh first.")
        return 1

    sns.set_theme(style="ticks", font_scale=1.05)
    produced = []

    for source, frame in all_df.groupby("source"):
        plotter = PLOTTERS.get(source)
        if plotter is None:
            print(f"  no plotter registered for '{source}', skipping")
            continue
        print(f"Plotting {source} ...")
        destination = os.path.join(output_root, source)
        plotter(frame, destination, produced)

    # Generate a summary dashboard combining throughput across all benchmarks.
    generate_summary_dashboard(all_df, output_root, produced)

    print("\nGenerated figures:")
    for path in produced:
        print(f"  {path}")
    if not produced:
        print("  (none)")
    return 0

def generate_summary_dashboard(all_df, output_root, produced):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # SAE gemm speedup summary
    gemm = all_df[all_df["source"] == "sae_gemm_benchmark"].copy()
    if not gemm.empty:
        gemm["backend"] = np.where(gemm["name"].str.contains("gemm", case=False), "GEMM", "Kernel")
        gemm["batch"] = gemm["name"].apply(extract_batch)
        for task, ax in zip(["Encoder", "Decoder"], [axes[0, 0], axes[0, 1]]):
            sub = gemm[gemm["name"].str.startswith(f"BM_{task}")]
            if sub.empty:
                continue
            pivot = sub.pivot_table(index="batch", columns="backend", values="latency_ms", aggfunc="mean")
            if {"Kernel", "GEMM"}.issubset(pivot.columns):
                ratio = pivot["Kernel"] / pivot["GEMM"]
                ax.bar(ratio.index.astype(str), ratio.values, color="#2ca02c", width=0.6)
                ax.axhline(1.0, color="red", linestyle="--", linewidth=1)
                ax.set_title(f"SAE {task}: GEMM speedup", fontsize=11, fontweight="bold")
                ax.set_xlabel("Batch size")
                ax.set_ylabel("Speedup (kernel/GEMM)\n(higher is better)")
                ax.grid(True, axis="y", ls="--", alpha=0.3)

    # Allocation bandwidth
    alloc = all_df[all_df["source"] == "allocation_benchmark"].copy()
    if not alloc.empty and "bytes_per_second" in alloc.columns:
        alloc["allocator"] = alloc["label"].where(alloc["label"] != "", "alloc")
        alloc["size"] = alloc["name"].apply(extract_int_arg).apply(human_bytes)
        bw = alloc.dropna(subset=["bytes_per_second"]).copy()
        bw["GB/s"] = bw["bytes_per_second"] / 1e9
        if not bw.empty:
            ax = axes[1, 0]
            sns.barplot(data=bw, x="size", y="GB/s", hue="allocator", ax=ax, width=0.7)
            ax.set_title("GPU allocation: bandwidth", fontsize=11, fontweight="bold")
            ax.set_xlabel("Allocation size")
            ax.set_ylabel("GB/s\n(higher is better)")
            ax.legend(title="", fontsize=8)
            ax.grid(True, axis="y", ls="--", alpha=0.3)

    # Pipeline throughput
    pipe = all_df[all_df["source"] == "pipeline_benchmark"].copy()
    if not pipe.empty and "msg/s" in pipe.columns:
        pipe["msg/s"] = pd.to_numeric(pipe["msg/s"], errors="coerce")
        pipe = pipe.dropna(subset=["msg/s"]).sort_values("msg/s", ascending=True)
        if not pipe.empty:
            ax = axes[1, 1]
            ax.barh(pipe["label"], pipe["msg/s"], color="#9467bd", height=0.65)
            ax.set_title("Pipeline: message throughput", fontsize=11, fontweight="bold")
            ax.set_xlabel("Messages / second\n(higher is better)")
            ax.set_ylabel("")
            ax.grid(True, axis="x", ls="--", alpha=0.3)

    fig.suptitle(f"Spire Benchmark Summary Dashboard\n{GPU_LABEL}", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(output_root, exist_ok=True)
    path = os.path.join(output_root, "summary_dashboard.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    produced.append(path)

if __name__ == "__main__":
    sys.exit(main())
