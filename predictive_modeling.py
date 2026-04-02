"""
Predictive modeling module for Nykaa Marketing Analytics.
Provides a PredictiveModeler class that app.py can import and use.
xgboost, catboost and shap are optional; the module degrades gracefully.
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from catboost import CatBoostRegressor
    CAT_AVAILABLE = True
except ImportError:
    CAT_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


# Columns to always drop before modelling
_DROP_COLS = ['campaign_id', 'channel_used', 'date', 'roi_category']


def _preprocess(df: pd.DataFrame, target: str):
    """Return (X, y) ready for sklearn."""
    df = df.copy()
    drop = [c for c in _DROP_COLS if c in df.columns]
    if target in ('revenue', 'conversions', 'leads'):
        drop += [c for c in ['roi', 'roas', 'profit', 'profit_margin', 'ctr',
                              'cpc', 'cpl', 'conversion_rate', 'engage_efficiency']
                 if c in df.columns and c != target]
    df = df.drop(columns=drop).fillna(0)

    obj_cols = [c for c in df.select_dtypes(include='object').columns
                if c != target]
    if obj_cols:
        df = pd.get_dummies(df, columns=obj_cols, drop_first=True)

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found after preprocessing.")

    X = df.drop(columns=[target])
    y = df[target]
    return X, y


class PredictiveModeler:
    """Trains and evaluates ML models for campaign outcome prediction."""

    AVAILABLE_MODELS = ['Random Forest', 'Gradient Boosting', 'Ridge Regression']

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.results = {}
        self.feature_importance = None
        if XGB_AVAILABLE:
            self.AVAILABLE_MODELS = self.AVAILABLE_MODELS + ['XGBoost']
        if CAT_AVAILABLE:
            self.AVAILABLE_MODELS = self.AVAILABLE_MODELS + ['CatBoost']

    def _evaluate(self, model, X_test, y_test) -> dict:
        preds = model.predict(X_test)
        return {
            'r2':          round(float(r2_score(y_test, preds)), 4),
            'rmse':        round(float(np.sqrt(mean_squared_error(y_test, preds))), 4),
            'mae':         round(float(mean_absolute_error(y_test, preds)), 4),
            'predictions': pd.Series(preds, index=y_test.index),
            'actual':      y_test.reset_index(drop=True),
        }

    def _feature_importance_df(self, model, feature_names) -> pd.DataFrame:
        try:
            imps = model.feature_importances_
            return (
                pd.DataFrame({'feature': feature_names, 'importance': imps})
                .sort_values('importance', ascending=False)
                .reset_index(drop=True)
            )
        except AttributeError:
            try:
                imps = np.abs(model.coef_)
                return (
                    pd.DataFrame({'feature': feature_names, 'importance': imps})
                    .sort_values('importance', ascending=False)
                    .reset_index(drop=True)
                )
            except Exception:
                return None

    def run_full_pipeline(self, target: str = 'revenue',
                          selected_models: list = None,
                          tune_models: bool = False) -> dict:
        """
        Train selected models, evaluate, and return results dict.
        """
        try:
            X, y = _preprocess(self.df, target)
        except ValueError as e:
            return {'error': str(e)}

        if len(X) < 10:
            return {'error': 'Not enough data to train models (need >= 10 rows).'}

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        if selected_models is None:
            selected_models = ['Random Forest']

        results = {}
        best_r2 = -np.inf
        best_model_name = None
        best_model_obj = None

        # ── Random Forest ────────────────────────────────────────────────────
        if 'Random Forest' in selected_models:
            if tune_models:
                params = {'n_estimators': [50, 100], 'max_depth': [5, 10, None]}
                grid = GridSearchCV(
                    RandomForestRegressor(random_state=42, n_jobs=-1),
                    params, cv=3, scoring='r2', n_jobs=-1)
                grid.fit(X_train, y_train)
                rf = grid.best_estimator_
            else:
                rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                rf.fit(X_train, y_train)
            results['Random Forest'] = self._evaluate(rf, X_test, y_test)
            if results['Random Forest']['r2'] > best_r2:
                best_r2, best_model_name, best_model_obj = results['Random Forest']['r2'], 'Random Forest', rf

        # ── Gradient Boosting ────────────────────────────────────────────────
        if 'Gradient Boosting' in selected_models:
            gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
            gb.fit(X_train, y_train)
            results['Gradient Boosting'] = self._evaluate(gb, X_test, y_test)
            if results['Gradient Boosting']['r2'] > best_r2:
                best_r2, best_model_name, best_model_obj = results['Gradient Boosting']['r2'], 'Gradient Boosting', gb

        # ── Ridge Regression ─────────────────────────────────────────────────
        if 'Ridge Regression' in selected_models:
            ridge = Ridge(alpha=1.0)
            ridge.fit(X_train, y_train)
            results['Ridge Regression'] = self._evaluate(ridge, X_test, y_test)
            if results['Ridge Regression']['r2'] > best_r2:
                best_r2, best_model_name, best_model_obj = results['Ridge Regression']['r2'], 'Ridge Regression', ridge

        # ── XGBoost ──────────────────────────────────────────────────────────
        if 'XGBoost' in selected_models and XGB_AVAILABLE:
            if tune_models:
                xgb_params = {'n_estimators': [50, 100], 'max_depth': [3, 6],
                              'learning_rate': [0.05, 0.1]}
                grid_xgb = GridSearchCV(
                    XGBRegressor(random_state=42, verbosity=0, n_jobs=-1),
                    xgb_params, cv=3, scoring='r2', n_jobs=-1)
                grid_xgb.fit(X_train, y_train)
                xgb = grid_xgb.best_estimator_
            else:
                xgb = XGBRegressor(n_estimators=100, learning_rate=0.1,
                                   random_state=42, verbosity=0, n_jobs=-1)
                xgb.fit(X_train, y_train)
            results['XGBoost'] = self._evaluate(xgb, X_test, y_test)
            if results['XGBoost']['r2'] > best_r2:
                best_r2, best_model_name, best_model_obj = results['XGBoost']['r2'], 'XGBoost', xgb

        # ── CatBoost ─────────────────────────────────────────────────────────
        if 'CatBoost' in selected_models and CAT_AVAILABLE:
            cat = CatBoostRegressor(iterations=100, learning_rate=0.1,
                                    depth=6, random_state=42, verbose=0)
            cat.fit(X_train, y_train)
            results['CatBoost'] = self._evaluate(cat, X_test, y_test)
            if results['CatBoost']['r2'] > best_r2:
                best_r2, best_model_name, best_model_obj = results['CatBoost']['r2'], 'CatBoost', cat

        if best_model_obj is not None:
            self.feature_importance = self._feature_importance_df(
                best_model_obj, list(X.columns))

        self.results = results
        return {
            'results':            results,
            'best_model':         best_model_name,
            'feature_importance': self.feature_importance,
        }


if __name__ == '__main__':
    import sys
    path   = sys.argv[1] if len(sys.argv) > 1 else 'processed_campaign_data.csv'
    target = sys.argv[2] if len(sys.argv) > 2 else 'revenue'
    df = pd.read_csv(path)
    modeler = PredictiveModeler(df)
    out = modeler.run_full_pipeline(target=target,
                                    selected_models=['Random Forest', 'Gradient Boosting'],
                                    tune_models=False)
    for name, m in out.get('results', {}).items():
        print(f"{name}: R2={m['r2']}  RMSE={m['rmse']}  MAE={m['mae']}")