import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from xgboost import XGBRegressor
import shap

# ─────────────────────────────────────────────
# GLOBAL STYLE
# ─────────────────────────────────────────────
PALETTE    = ["#6C63FF", "#00D4AA", "#FF6B9D", "#FFB347", "#64DFDF"]
BG         = "#0D0F1A"
CARD       = "#141728"
TEXT       = "#F0F2FF"
TEXT_MUTED = "#8A8FAD"
GRID       = "#252840"

rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor":   CARD,
    "axes.edgecolor":   GRID,
    "axes.labelcolor":  TEXT,
    "axes.titlecolor":  "#6C63FF",
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
    "xtick.color":      TEXT_MUTED,
    "ytick.color":      TEXT_MUTED,
    "text.color":       TEXT,
    "grid.color":       GRID,
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
    "font.family":      "monospace",
    "legend.facecolor": CARD,
    "legend.edgecolor": GRID,
})

DROP_COLS = ["Campaign_ID", "Channel"]


# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def preprocess(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    df = df.copy().drop(
        columns=[c for c in DROP_COLS if c in df.columns]
    ).fillna(0)

    # One-hot encode remaining object columns (excluding target)
    obj_cols = [c for c in df.select_dtypes(include="object").columns if c != target]
    if obj_cols:
        df = pd.get_dummies(df, columns=obj_cols, drop_first=True)

    X = df.drop(columns=[target])
    y = df[target]
    return X, y


# ─────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────
def train_random_forest(X: pd.DataFrame, y: pd.Series) -> RandomForestRegressor:
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)
    return model


def train_xgboost(X: pd.DataFrame, y: pd.Series) -> XGBRegressor:
    model = XGBRegressor(
        n_estimators=100, learning_rate=0.1,
        random_state=42, verbosity=0, n_jobs=-1,
    )
    model.fit(X, y)
    return model


def tune_random_forest(X: pd.DataFrame, y: pd.Series) -> RandomForestRegressor:
    params = {"n_estimators": [50, 100], "max_depth": [5, 10, None]}
    grid = GridSearchCV(RandomForestRegressor(random_state=42, n_jobs=-1),
                        params, cv=3, scoring="r2", n_jobs=-1)
    grid.fit(X, y)
    print("  Best RF params:", grid.best_params_)
    return grid.best_estimator_


def tune_xgboost(X: pd.DataFrame, y: pd.Series) -> XGBRegressor:
    params = {
        "n_estimators": [50, 100],
        "max_depth": [3, 6],
        "learning_rate": [0.05, 0.1],
    }
    grid = GridSearchCV(XGBRegressor(random_state=42, verbosity=0, n_jobs=-1),
                        params, cv=3, scoring="r2", n_jobs=-1)
    grid.fit(X, y)
    print("  Best XGB params:", grid.best_params_)
    return grid.best_estimator_


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────
def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series,
             name: str = "Model") -> dict:
    preds = model.predict(X_test)
    metrics = {
        "RMSE": round(np.sqrt(mean_squared_error(y_test, preds)), 4),
        "MAE":  round(mean_absolute_error(y_test, preds), 4),
        "R²":   round(r2_score(y_test, preds), 4),
    }
    print(f"\n  {name}")
    for k, v in metrics.items():
        print(f"    {k}: {v}")
    return {"name": name, "preds": preds, **metrics}


def plot_results(results: list[dict], y_test: pd.Series) -> None:
    """Actual vs Predicted scatter + metric comparison bar chart."""
    n = len(results)
    fig = plt.figure(figsize=(8 * n, 5))
    fig.suptitle("Model Evaluation", fontsize=15, color="#6C63FF", fontweight="bold")
    gs = gridspec.GridSpec(1, n + 1, width_ratios=[3] * n + [2])

    for i, res in enumerate(results):
        ax = fig.add_subplot(gs[i])
        ax.scatter(y_test, res["preds"],
                   color=PALETTE[i % len(PALETTE)], alpha=0.55, s=20, edgecolors="none")
        lims = [min(y_test.min(), res["preds"].min()),
                max(y_test.max(), res["preds"].max())]
        ax.plot(lims, lims, "--", color="#FFB347", linewidth=1.5, label="Perfect fit")
        ax.set_title(res["name"])
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.legend(fontsize=8)
        ax.text(0.05, 0.92, f"R² = {res['R²']}", transform=ax.transAxes,
                color=PALETTE[i % len(PALETTE)], fontsize=10)

    # Metric bar chart
    ax_bar = fig.add_subplot(gs[-1])
    metrics_to_plot = ["R²", "RMSE"]
    x = np.arange(len(metrics_to_plot))
    width = 0.8 / n
    for i, res in enumerate(results):
        vals = [res[m] for m in metrics_to_plot]
        ax_bar.bar(x + i * width, vals, width, label=res["name"],
                   color=PALETTE[i % len(PALETTE)], edgecolor="none", alpha=0.85)
    ax_bar.set_xticks(x + width * (n - 1) / 2)
    ax_bar.set_xticklabels(metrics_to_plot)
    ax_bar.set_title("Metric Comparison")
    ax_bar.legend(fontsize=8)

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# SHAP EXPLAINABILITY
# ─────────────────────────────────────────────
def explain_model(model, X: pd.DataFrame, title: str = "XGBoost") -> None:
    print(f"\n  Computing SHAP values for {title}…")
    explainer   = shap.Explainer(model, X)
    shap_values = explainer(X)

    shap.plots.beeswarm(shap_values, max_display=15, show=False)
    plt.gcf().suptitle(f"SHAP Beeswarm — {title}", color="#6C63FF",
                       fontweight="bold", fontsize=14)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# FULL PIPELINE
# ─────────────────────────────────────────────
def run_ml_pipeline(path: str, target: str = "Revenue") -> None:
    print("=" * 60)
    print("  ML PIPELINE — PREDICTIVE MODELING")
    print("=" * 60)

    df = load_data(path)
    X, y = preprocess(df, target)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("\n[1/4] Training baseline models…")
    rf  = train_random_forest(X_train, y_train)
    xgb = train_xgboost(X_train, y_train)

    res_rf  = evaluate(rf,  X_test, y_test, "Random Forest (baseline)")
    res_xgb = evaluate(xgb, X_test, y_test, "XGBoost (baseline)")

    print("\n[2/4] Tuning models via GridSearchCV…")
    rf_best  = tune_random_forest(X_train, y_train)
    xgb_best = tune_xgboost(X_train, y_train)

    res_rf_t  = evaluate(rf_best,  X_test, y_test, "Random Forest (tuned)")
    res_xgb_t = evaluate(xgb_best, X_test, y_test, "XGBoost (tuned)")

    print("\n[3/4] Plotting results…")
    plot_results([res_rf, res_xgb, res_rf_t, res_xgb_t], y_test)

    print("\n[4/4] SHAP explainability (tuned XGBoost)…")
    explain_model(xgb_best, X_test, title="XGBoost (tuned)")

    print("\n✅ Pipeline complete.")


if __name__ == "__main__":
    run_ml_pipeline("processed_campaign_data.csv", target="Revenue")