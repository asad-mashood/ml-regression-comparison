"""All figures for the report. Each model keeps the same colour in every chart."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from .models import MODEL_ORDER

COLORS = dict(zip(MODEL_ORDER, [
    "#2a78d6",  # blue    - Linear Regression
    "#eb6834",  # orange  - Decision Tree
    "#1baf7a",  # aqua    - Random Forest
    "#eda100",  # yellow  - Neural Network
    "#e87ba4",  # magenta - XGBoost
    "#008300",  # green   - KNN
]))
INK, INK2, GRID = "#1f1f1e", "#5f5e58", "#e6e5e0"

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
})


def _k(x, _pos=None):
    return f"${x/1000:,.0f}k"


def plot_actual_vs_predicted(y_test, predictions: dict, metrics: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5), sharex=True, sharey=True)
    lo, hi = 0, 520_000
    for ax, name in zip(axes.ravel(), MODEL_ORDER):
        ax.scatter(y_test, predictions[name], s=8, alpha=0.35, color=COLORS[name],
                   edgecolors="none")
        ax.plot([lo, hi], [lo, hi], color=INK, lw=1.2, ls="--", label="Perfect prediction")
        r = metrics.loc[name]
        ax.set_title(name)
        ax.text(0.03, 0.97, f"R² = {r.R2:.3f}\nRMSE = ${r.RMSE:,.0f}\nMAE = ${r.MAE:,.0f}",
                transform=ax.transAxes, va="top", fontsize=9, color=INK,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=GRID))
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.xaxis.set_major_formatter(_k); ax.yaxis.set_major_formatter(_k)
    for ax in axes[1]:
        ax.set_xlabel("Actual median house value")
    for ax in axes[:, 0]:
        ax.set_ylabel("Predicted median house value")
    axes[0, 0].legend(loc="lower right", frameon=False, fontsize=9)
    fig.suptitle("Actual vs predicted house values on the 20% test set (n = 4,128)",
                 fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(out); plt.close(fig)


def plot_metric_bars(metrics: pd.DataFrame, out: Path):
    panels = [("MAE", "MAE (USD) — lower is better", True),
              ("RMSE", "RMSE (USD) — lower is better", True),
              ("MSE", "MSE (USD²) — lower is better", True),
              ("R2", "R² score — higher is better", False)]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    for ax, (col, title, money) in zip(axes.ravel(), panels):
        vals = metrics.loc[MODEL_ORDER, col]
        y = np.arange(len(MODEL_ORDER))[::-1]
        ax.barh(y, vals, color=[COLORS[m] for m in MODEL_ORDER], height=0.62)
        ax.set_yticks(y, MODEL_ORDER)
        ax.set_title(title, loc="left")
        ax.grid(axis="y", visible=False)
        vmax = vals.max()
        for yi, v in zip(y, vals):
            label = (f"{v/1e9:.2f}B" if col == "MSE" else f"${v:,.0f}") if money else f"{v:.3f}"
            ax.text(v + vmax * 0.01, yi, label, va="center", fontsize=9, color=INK)
        ax.set_xlim(0, vmax * 1.18 if money else 1.0)
        if col == "MSE":
            ax.xaxis.set_major_formatter(lambda x, _: f"{x/1e9:.0f}B")
        elif money:
            ax.xaxis.set_major_formatter(_k)
    fig.suptitle("Test-set performance of the six regression models",
                 fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(out); plt.close(fig)


def plot_residuals(y_test, predictions: dict, out: Path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    bins = np.linspace(-250_000, 250_000, 61)
    for ax, name in zip(axes.ravel(), MODEL_ORDER):
        res = np.asarray(y_test) - predictions[name]
        ax.hist(res, bins=bins, color=COLORS[name], alpha=0.9, edgecolor="white", linewidth=0.5)
        ax.axvline(0, color=INK, lw=1.1, ls="--")
        ax.set_title(f"{name}  (mean {res.mean():+,.0f}, sd {res.std():,.0f})")
        ax.xaxis.set_major_formatter(_k)
    for ax in axes[1]:
        ax.set_xlabel("Residual = actual − predicted")
    for ax in axes[:, 0]:
        ax.set_ylabel("Number of test districts")
    fig.suptitle("Residual distributions on the test set (narrower and centred on 0 is better)",
                 fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(out); plt.close(fig)


def plot_sample_predictions(y_test, predictions: dict, out: Path, n: int = 40, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(y_test), n, replace=False))
    x = np.arange(n)
    y_true = np.asarray(y_test)[idx]
    order = np.argsort(y_true)
    fig, ax = plt.subplots(figsize=(15, 5.5))
    ax.plot(x, y_true[order], color=INK, lw=2.4, marker="o", ms=5, label="Actual", zorder=5)
    for name in MODEL_ORDER:
        ax.plot(x, predictions[name][idx][order], color=COLORS[name], lw=1.4,
                marker="o", ms=3.5, alpha=0.9, label=name)
    ax.set_xlabel(f"{n} randomly chosen test districts (sorted by actual value)")
    ax.set_ylabel("Median house value")
    ax.yaxis.set_major_formatter(_k)
    ax.legend(ncol=7, loc="upper left", frameon=False, fontsize=9)
    ax.set_title("Actual vs predicted values for individual test districts", loc="left")
    fig.tight_layout()
    fig.savefig(out); plt.close(fig)


def plot_feature_importance(importances: dict, out: Path, top: int = 12):
    fig, axes = plt.subplots(1, len(importances), figsize=(14, 5.5))
    for ax, (name, s) in zip(np.atleast_1d(axes), importances.items()):
        s = s.sort_values().tail(top)
        ax.barh(s.index, s.values, color=COLORS[name], height=0.62)
        ax.set_title(f"{name} — feature importance", loc="left")
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out); plt.close(fig)


def plot_knn_k(ks, scores, best_k, out: Path):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(ks, scores, color=COLORS["KNN"], lw=2, marker="o", ms=5)
    ax.axvline(best_k, color=INK2, ls="--", lw=1)
    ax.annotate(f"best k = {best_k}", (best_k, max(scores)), xytext=(8, -14),
                textcoords="offset points", color=INK)
    ax.set_xlabel("Number of neighbours k")
    ax.set_ylabel("5-fold CV RMSE on training set (USD)")
    ax.yaxis.set_major_formatter(_k)
    ax.set_title("Choosing k for KNN (training data only)", loc="left")
    fig.tight_layout()
    fig.savefig(out); plt.close(fig)


def plot_eda(df: pd.DataFrame, out_corr: Path, out_target: Path):
    num = df.select_dtypes("number")
    corr = num.corr()
    cmap = LinearSegmentedColormap.from_list("div", ["#eb6834", "#efeee9", "#2a78d6"])
    fig, ax = plt.subplots(figsize=(10, 8.5))
    im = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)), corr.columns)
    ax.grid(False)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7.5, color=INK)
    fig.colorbar(im, ax=ax, fraction=0.04, label="Pearson correlation")
    ax.set_title("Correlation matrix of numeric features", loc="left")
    fig.tight_layout(); fig.savefig(out_corr); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].hist(df["median_house_value"], bins=60, color=COLORS["Linear Regression"],
                 edgecolor="white", linewidth=0.5)
    axes[0].xaxis.set_major_formatter(_k)
    axes[0].set_title("Target: median_house_value (note the cap at $500k)", loc="left")
    axes[0].set_xlabel("Median house value"); axes[0].set_ylabel("Districts")
    axes[1].scatter(df["median_income"], df["median_house_value"], s=4, alpha=0.2,
                    color=COLORS["Linear Regression"], edgecolors="none")
    axes[1].yaxis.set_major_formatter(_k)
    axes[1].set_title("Strongest predictor: median_income", loc="left")
    axes[1].set_xlabel("Median income (tens of thousands USD)"); axes[1].set_ylabel("Median house value")
    fig.tight_layout(); fig.savefig(out_target); plt.close(fig)
