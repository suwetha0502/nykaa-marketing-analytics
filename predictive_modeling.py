import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from xgboost import XGBRegressor
import shap


# ---------------- LOAD DATA ----------------
def load_data(path):
    return pd.read_csv(path)


# ---------------- PREPROCESS ----------------
def preprocess(df, target):
    df = df.copy()

    # Drop non-useful columns
    drop_cols = ["Campaign_ID", "Channel"]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    # Fill NA
    df = df.fillna(0)

    # Features & target
    X = df.drop(columns=[target])
    y = df[target]

    return X, y


# ---------------- TRAIN MODELS ----------------
def train_random_forest(X, y):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model


def train_xgboost(X, y):
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X, y)
    return model


# ---------------- HYPERPARAM TUNING ----------------
def tune_random_forest(X, y):
    params = {
        "n_estimators": [50, 100],
        "max_depth": [5, 10, None]
    }

    grid = GridSearchCV(RandomForestRegressor(random_state=42),
                        params, cv=3, scoring="r2")
    grid.fit(X, y)

    print("Best RF Params:", grid.best_params_)
    return grid.best_estimator_


def tune_xgboost(X, y):
    params = {
        "n_estimators": [50, 100],
        "max_depth": [3, 6],
        "learning_rate": [0.05, 0.1]
    }

    grid = GridSearchCV(XGBRegressor(random_state=42),
                        params, cv=3, scoring="r2")
    grid.fit(X, y)

    print("Best XGB Params:", grid.best_params_)
    return grid.best_estimator_


# ---------------- EVALUATION ----------------
def evaluate(model, X_test, y_test, name="Model"):
    preds = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print(f"\n{name} Performance:")
    print("RMSE:", round(rmse, 2))
    print("R2:", round(r2, 2))


# ---------------- SHAP EXPLAINABILITY ----------------
def explain_model(model, X):
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    shap.plots.beeswarm(shap_values)


# ---------------- MAIN PIPELINE ----------------
def run_ml_pipeline(path, target="Revenue"):
    df = load_data(path)

    X, y = preprocess(df, target)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train models
    rf = train_random_forest(X_train, y_train)
    xgb = train_xgboost(X_train, y_train)

    # Evaluate
    evaluate(rf, X_test, y_test, "Random Forest")
    evaluate(xgb, X_test, y_test, "XGBoost")

    # Tune models
    rf_best = tune_random_forest(X_train, y_train)
    xgb_best = tune_xgboost(X_train, y_train)

    # Evaluate tuned models
    evaluate(rf_best, X_test, y_test, "Tuned RF")
    evaluate(xgb_best, X_test, y_test, "Tuned XGB")

    # Explain best model (XGBoost)
    explain_model(xgb_best, X_test)


# ---------------- RUN ----------------
if __name__ == "__main__":
    run_ml_pipeline("processed_campaign_data.csv", target="Revenue")