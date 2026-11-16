"""
Comparative implementation of six regression algorithms on one dataset.

Same data, same features, same target, same 80:20 split for every model,
so the comparison is fair. Metrics: MAE, MSE, RMSE, R2.

Run:
    python regression_comparison.py                      # Diabetes dataset (bundled with scikit-learn, works offline)
    python regression_comparison.py --dataset california # California house prices (downloads once)
"""
import argparse
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

try:
    from xgboost import XGBRegressor
    HAVE_XGB = True
except ImportError:  # pip install xgboost
    HAVE_XGB = False

SEED = 42
OUT = pathlib.Path(__file__).parent / "results"


def load(name):
    if name == "california":
        d = fetch_california_housing(as_frame=True)
        return d.data, d.target, "California Housing", "Median house value ($100k)"
    d = load_diabetes(as_frame=True, scaled=False)
    return d.data, d.target, "Diabetes (scikit-learn)", "Disease progression score"


def build_models():
    """Scaling is applied only where the algorithm needs it (distance or gradient based)."""
    if HAVE_XGB:
        boost_name = "XGBoost"
        boost = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=3,
                             subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED)
    else:
        boost_name = "Gradient Boosting (sklearn, XGBoost not installed)"
        boost = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3,
                                          subsample=0.8, random_state=SEED)
    return {
        "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Decision Tree": DecisionTreeRegressor(max_depth=4, min_samples_leaf=10, random_state=SEED),
        "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=SEED),
        "Neural Network": make_pipeline(StandardScaler(), MLPRegressor(
            hidden_layer_sizes=(32, 16), alpha=1.0, learning_rate_init=0.01,
            max_iter=3000, early_stopping=True, n_iter_no_change=50, random_state=SEED)),
        boost_name: boost,
        "KNN": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=15, weights="distance")),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["diabetes", "california"], default="diabetes")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    X, y, ds_name, target_name = load(args.dataset)
    X = X.dropna()
    y = y.loc[X.index]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=SEED)
    print(f"Dataset: {ds_name} | rows={len(X)} features={X.shape[1]} | train={len(X_tr)} test={len(X_te)}")

    rows, preds = [], {}
    for name, model in build_models().items():
        model.fit(X_tr, y_tr)
        p = model.predict(X_te)
        preds[name] = p
        mse = mean_squared_error(y_te, p)
        rows.append({"Model": name, "MAE": mean_absolute_error(y_te, p), "MSE": mse,
                     "RMSE": np.sqrt(mse), "R2": r2_score(y_te, p)})
    res = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    res.to_csv(OUT / "metrics.csv", index=False)
    table = res.copy()
    for c in ["MAE", "MSE", "RMSE"]:
        table[c] = table[c].map(lambda v: f"{v:.2f}")
    table["R2"] = table["R2"].map(lambda v: f"{v:.3f}")
    md = "| " + " | ".join(table.columns) + " |\n|" + "---|" * len(table.columns) + "\n"
    md += "\n".join("| " + " | ".join(r) + " |" for r in table.values.tolist()) + "\n"
    (OUT / "metrics.md").write_text(f"Dataset: {ds_name}, 80:20 split, random_state={SEED}\n\n" + md)
    print(md)
    print("Best by RMSE:", res.iloc[0]["Model"])

    # Actual vs predicted, one panel per model
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    lo, hi = float(y_te.min()), float(y_te.max())
    for ax, (name, p) in zip(axes.ravel(), preds.items()):
        ax.scatter(y_te, p, s=18, alpha=0.7)
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        r = res.set_index("Model").loc[name]
        ax.set_title(f"{name.split(' (')[0]}\nRMSE {r.RMSE:.2f} | R2 {r.R2:.3f}", fontsize=11)
        ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
    fig.suptitle(f"Actual vs predicted on the test set: {ds_name}", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "actual_vs_predicted.png", dpi=140)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    short = [m.split(" (")[0] for m in res["Model"]]
    ax.barh(short[::-1], res["RMSE"][::-1])
    ax.set_xlabel("RMSE on test set (lower is better)")
    ax.set_title(f"Model comparison: {ds_name}")
    fig.tight_layout()
    fig.savefig(OUT / "rmse_comparison.png", dpi=140)

    pd.DataFrame({"actual": y_te.values, **{k.split(" (")[0]: v for k, v in preds.items()}}).to_csv(
        OUT / "test_predictions.csv", index=False)


if __name__ == "__main__":
    main()
