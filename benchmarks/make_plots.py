"""
Plots for the penalized Saito validation study and experiment reruns.

Usage:
    python benchmarks/make_plots.py --bench <benchmark_dir> \
        [--ext <extension_dir>] [--rl <rl_dir>] --out <plots_dir>

Reads only the machine-readable JSON outputs; writes PNGs.
"""

import argparse
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# validated reference palette (dataviz skill, light mode, fixed slot order)
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3e0"
FREE_C, NONFREE_C = BLUE, ORANGE
LOSS_FLOOR = 1e-16

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True,
    "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "lines.linewidth": 2.0, "legend.frameon": False,
})


def _load(path):
    with open(path) as f:
        return json.load(f)


def _save(fig, out, name):
    fig.tight_layout()
    fig.savefig(os.path.join(out, name), dpi=150)
    plt.close(fig)
    print(f"  wrote {name}")


def _floor(v):
    return max(float(v), LOSS_FLOOR)


def plot_lambda_sweep(bench, out):
    rows = _load(os.path.join(bench, "lambda_sweep.json"))
    series = defaultdict(list)
    labels = {}
    for r in rows:
        series[r["name"]].append((r["lambda"], _floor(r["loss"])))
        labels[r["name"]] = r["label"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, pts in series.items():
        pts.sort()
        xs, ys = zip(*pts)
        c = FREE_C if labels[name] == "free" else NONFREE_C
        ax.plot(xs, ys, color=c, alpha=0.55, marker="o", markersize=3.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("penalty weight λ")
    ax.set_ylabel("penalized Saito loss (floored at 1e-16)")
    ax.set_title("λ sweep — free arrangements stay at 0; nonfree rise toward 1")
    ax.plot([], [], color=FREE_C, label="free (exactly certified)")
    ax.plot([], [], color=NONFREE_C, label="nonfree (exactly verified)")
    ax.legend(loc="center left")
    _save(fig, out, "lambda_sweep.png")


def plot_beta_sweep(bench, out):
    rows = _load(os.path.join(bench, "beta_sweep.json"))
    series = defaultdict(list)
    labels = {}
    for r in rows:
        series[r["name"]].append((r["beta"], _floor(r["loss"])))
        labels[r["name"]] = r["label"]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for name, pts in series.items():
        pts.sort()
        xs, ys = zip(*pts)
        c = FREE_C if labels[name] == "free" else NONFREE_C
        ax.plot(xs, ys, color=c, alpha=0.55, marker="o", markersize=3.5)
    ax.set_yscale("log")
    ax.set_xlabel("residual exponent β")
    ax.set_ylabel("penalized Saito loss (floored)")
    ax.set_title("β sweep (λ = 1)")
    ax.plot([], [], color=FREE_C, label="free")
    ax.plot([], [], color=NONFREE_C, label="nonfree")
    ax.legend()
    _save(fig, out, "beta_sweep.png")


def plot_perturbation(bench, out):
    rows = _load(os.path.join(bench, "perturbation.json"))
    ts, news, legs = [], [], []
    for r in rows:
        tag = r["tag"]
        if tag == "t=0":
            continue
        t = 1.0 / float(tag.split("/")[1])
        ts.append(t)
        news.append(_floor(r["loss"]))
        legs.append(_floor(r["legacy_score"]))
    order = np.argsort(ts)
    ts = np.array(ts)[order]
    news = np.array(news)[order]
    legs = np.array(legs)[order]
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.plot(ts, news, color=BLUE, marker="o", label="penalized loss (new)")
    ax.plot(ts, legs, color=ORANGE, marker="s",
            label="legacy angular score (invalid)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("perturbation size t of one line of a free arrangement")
    ax.set_ylabel("score (floored at 1e-16)")
    ax.set_title("Degeneration path: computed loss becomes small along the\n"
                 "tested path; legacy score jumps between ~1 and ~0")
    ax.legend()
    _save(fig, out, "perturbation.png")


def plot_old_vs_new(bench, out):
    rows = _load(os.path.join(bench, "main_table.json"))
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    for r in rows:
        c = FREE_C if r["label"] == "free" else NONFREE_C
        ax.scatter(_floor(r["legacy_score"]), _floor(r["loss"]),
                   s=55, color=c, edgecolors="white", linewidths=1.2,
                   zorder=3)
    ax.axvline(0.05, color=INK2, linewidth=1, linestyle="--")
    ax.text(0.05, 2e-16, " old 0.05 threshold", fontsize=8, color=INK2,
            rotation=90, va="bottom")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("legacy angular score (mathematically binary; regression only)")
    ax.set_ylabel("penalized Saito loss (new)")
    ax.set_title("Benchmark: new loss separates free/nonfree;\n"
                 "legacy values scatter across 16 decades")
    ax.scatter([], [], color=FREE_C, label="free (certified)")
    ax.scatter([], [], color=NONFREE_C, label="nonfree (verified)")
    ax.legend(loc="center left")
    _save(fig, out, "old_vs_new_scatter.png")


def plot_restart_study(bench, out):
    rows = _load(os.path.join(bench, "restart_study.json"))
    agg = defaultdict(lambda: defaultdict(list))
    labels = {}
    for r in rows:
        agg[r["name"]][r["n_restarts"]].append(_floor(r["loss"]))
        labels[r["name"]] = r["label"]
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for name, by_nr in agg.items():
        xs = sorted(by_nr)
        med = [np.median(by_nr[k]) for k in xs]
        lo = [np.min(by_nr[k]) for k in xs]
        hi = [np.max(by_nr[k]) for k in xs]
        c = FREE_C if labels[name] == "free" else NONFREE_C
        ax.plot(xs, med, color=c, alpha=0.6, marker="o", markersize=3.5)
        ax.fill_between(xs, lo, hi, color=c, alpha=0.12, linewidth=0)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("number of restarts")
    ax.set_ylabel("loss (median over seeds; band = min–max)")
    ax.set_title("Restart-count study")
    ax.plot([], [], color=FREE_C, label="free")
    ax.plot([], [], color=NONFREE_C, label="nonfree")
    ax.legend()
    _save(fig, out, "restart_study.png")


def plot_iteration_study(bench, out):
    rows = _load(os.path.join(bench, "iteration_study.json"))
    series = defaultdict(list)
    labels = {}
    for r in rows:
        series[r["name"]].append((r["n_iters"], _floor(r["loss"])))
        labels[r["name"]] = r["label"]
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for name, pts in series.items():
        pts.sort()
        xs, ys = zip(*pts)
        c = FREE_C if labels[name] == "free" else NONFREE_C
        ax.plot(xs, ys, color=c, alpha=0.6, marker="o", markersize=3.5)
    ax.set_yscale("log")
    ax.set_xlabel("MM sweeps per restart")
    ax.set_ylabel("loss (floored)")
    ax.set_title("Iteration-budget study")
    ax.plot([], [], color=FREE_C, label="free")
    ax.plot([], [], color=NONFREE_C, label="nonfree")
    ax.legend()
    _save(fig, out, "iteration_study.png")


def plot_components(bench, out):
    rows = _load(os.path.join(bench, "lambda_sweep.json"))
    target = "nonfree7_a"
    pts = sorted((r["lambda"], r["parts"]) for r in rows
                 if r["name"] == target)
    if not pts:
        return
    lams = [p[0] for p in pts]
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for key, c, label in (("inner_abs", BLUE, "|<B,q>| (alignment)"),
                          ("B_norm", AQUA, "||B|| (determinant size)"),
                          ("L1u_norm", YELLOW, "||L1 u|| (tangency residual, u)"),
                          ("L2v_norm", MAGENTA, "||L2 v|| (tangency residual, v)")):
        ax.plot(lams, [max(p[1][key], LOSS_FLOOR) for p in pts], color=c,
                marker="o", markersize=3.5, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("penalty weight λ")
    ax.set_ylabel("component value at the optimum")
    ax.set_title(f"Objective components vs λ ({target})")
    ax.legend(fontsize=8)
    _save(fig, out, "components_lambda.png")


def plot_extension(ext, out):
    rows = _load(os.path.join(ext, "extension_rows.json"))
    report = _load(os.path.join(ext, "extension_report.json"))
    all_rows = rows["validation"] + rows["test"]
    free = [max(r["new_loss"], LOSS_FLOOR) for r in all_rows if r["is_free"]]
    nonf = [max(r["new_loss"], LOSS_FLOOR) for r in all_rows
            if not r["is_free"]]
    bins = np.logspace(-16, 0, 33)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.hist(nonf, bins=bins, color=NONFREE_C, alpha=0.75,
            label=f"nonfree extensions (n={len(nonf)})")
    ax.hist(free, bins=bins, color=FREE_C, alpha=0.75,
            label=f"free extensions (n={len(free)})")
    tau = report.get("refit_threshold_new")
    if tau:
        ax.axvline(tau, color=INK, linewidth=1.4, linestyle="--")
        ax.text(tau, ax.get_ylim()[1] * 0.9, f" refit τ = {tau:.2e}",
                fontsize=8, color=INK)
    ax.axvline(0.05, color=INK2, linewidth=1, linestyle=":")
    ax.text(0.05, ax.get_ylim()[1] * 0.72, " old 0.05", fontsize=8,
            color=INK2)
    ax.set_xscale("log")
    ax.set_xlabel("penalized Saito loss of the one-line extension")
    ax.set_ylabel("count")
    ax.set_title("Extension pre-filter: loss distribution by exact label")
    ax.legend()
    _save(fig, out, "extension_filter.png")


def plot_rl(rl_dir, out):
    # aggregate the per-arm summaries produced by run_rl_comparison.py
    arms, results = [], []
    for arm in sorted(os.listdir(rl_dir)):
        p = os.path.join(rl_dir, arm, "rl_comparison_summary.json")
        if os.path.isdir(os.path.join(rl_dir, arm)) and os.path.exists(p):
            s = _load(p)
            for name, v in s.items():
                arms.append(name)
                results.append(v)
    if not arms:
        print("  (no RL summaries found; skipping RL plots)")
        return
    order = ["penalized", "potential", "combinatorial", "terminal",
             "random", "legacy"]
    pairs = sorted(zip(arms, results),
                   key=lambda t: order.index(t[0]) if t[0] in order else 99)
    arms = [p[0] for p in pairs]
    results = [p[1] for p in pairs]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    xs = np.arange(len(arms))
    means = [r["certified_mean"] for r in results]
    ax.bar(xs, means, width=0.62, color=BLUE, edgecolor="white", zorder=3)
    for i, r in enumerate(results):
        vals = r["certified_all"]
        ax.scatter([i] * len(vals) +
                   np.linspace(-0.12, 0.12, len(vals)), vals,
                   color=INK, s=14, zorder=4)
    ax.set_xticks(xs, arms, rotation=15)
    ax.set_ylabel("exactly certified free arrangements per seed")
    ax.set_title("RL arms at equal step budget (dots = individual seeds)")
    _save(fig, out, "rl_certified_by_arm.png")

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    colors = [BLUE, AQUA, YELLOW, MAGENTA, GREEN, ORANGE]
    for (arm, _v), c in zip(pairs, colors):
        comp = _load(os.path.join(rl_dir, arm, "rl_comparison.json"))
        by_steps = defaultdict(list)
        for run in comp:
            for pt in run["curve"]:
                by_steps[pt["steps"]].append(pt["free_found"])
        xs = sorted(by_steps)
        mean = [np.mean(by_steps[s]) for s in xs]
        lo = [np.min(by_steps[s]) for s in xs]
        hi = [np.max(by_steps[s]) for s in xs]
        ax.plot(xs, mean, color=c, label=arm)
        ax.fill_between(xs, lo, hi, color=c, alpha=0.10, linewidth=0)
    ax.set_xlabel("environment steps")
    ax.set_ylabel("cumulative free arrangements (training-time count)")
    ax.set_title("Discovery curves by reward arm (mean over seeds; "
                 "band = min–max)")
    ax.legend(fontsize=8)
    _save(fig, out, "rl_discovery_curves.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--ext", default=None)
    ap.add_argument("--rl", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    plot_lambda_sweep(args.bench, args.out)
    plot_beta_sweep(args.bench, args.out)
    plot_perturbation(args.bench, args.out)
    plot_old_vs_new(args.bench, args.out)
    plot_restart_study(args.bench, args.out)
    plot_iteration_study(args.bench, args.out)
    plot_components(args.bench, args.out)
    if args.ext and os.path.exists(os.path.join(args.ext,
                                                "extension_rows.json")):
        plot_extension(args.ext, args.out)
    if args.rl and os.path.isdir(args.rl):
        plot_rl(args.rl, args.out)


if __name__ == "__main__":
    main()
