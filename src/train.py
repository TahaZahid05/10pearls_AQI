from pathlib import Path
import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
import shap
import dagshub
import mlflow
import mlflow.sklearn

from src.config import HISTORICAL_PARQUET, BASE_DIR, TARGET_COLS
from src.feature_engineering import get_feature_and_target_columns

MODELS_DIR = BASE_DIR / "models"
PLOTS_DIR = BASE_DIR / "plots"
MODELS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

DAGSHUB_OWNER = "TahaZahid05"
DAGSHUB_REPO = "10pearls_AQI"
MODEL_REGISTRY_NAME = "AQI_Predictor_Model"


def init_mlflow():
    dagshub.init(repo_owner=DAGSHUB_OWNER, repo_name=DAGSHUB_REPO, mlflow=True)
    mlflow.set_experiment("AQI_3Day_Forecasting")


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, horizon_name: str) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {
        f"{horizon_name}_mae": float(mae),
        f"{horizon_name}_rmse": float(rmse),
        f"{horizon_name}_r2": float(r2),
    }


def train_and_evaluate_all():
    print("Starting Model Training & Evaluation Pipeline")

    init_mlflow()

    # 1. Load Data
    print(f"Loading historical dataset from {HISTORICAL_PARQUET}...")
    df = pd.read_parquet(HISTORICAL_PARQUET)
    feature_cols, target_cols = get_feature_and_target_columns(df)

    X = df[feature_cols]
    y = df[target_cols]

    split_idx = int(len(df) * 0.80)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"Training samples: {len(X_train)} | Test samples: {len(X_test)}")
    print(f"Input Features (X): {len(feature_cols)} | Targets (y): {len(target_cols)}")

    model_factories = {
        "Ridge_Regression": lambda: Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=1.0))
        ]),
        "Random_Forest": lambda: RandomForestRegressor(
            n_estimators=100, max_depth=12, random_state=42, n_jobs=-1
        ),
        "LightGBM": lambda: lgb.LGBMRegressor(
            n_estimators=150, max_depth=8, learning_rate=0.05, random_state=42, verbose=-1
        ),
        "Neural_Network_MLP": lambda: Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", MLPRegressor(
                hidden_layer_sizes=(64, 32), max_iter=200, random_state=42, early_stopping=True
            ))
        ])
    }

    results = {}
    fitted_models = {}

    for model_name, factory in model_factories.items():
        print(f"\nTraining [{model_name}]...")
        horizon_models = {}
        model_metrics = {}
        all_y_true = []
        all_y_pred = []

        with mlflow.start_run(run_name=model_name):
            mlflow.log_param("model_family", model_name)
            mlflow.log_param("num_features", len(feature_cols))
            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("test_samples", len(X_test))

            for horizon_col in target_cols:
                h_name = horizon_col.replace("target_aqi_", "")
                model = factory()
                model.fit(X_train, y_train[horizon_col])
                preds = model.predict(X_test)

                horizon_models[horizon_col] = model

                metrics = evaluate_predictions(y_test[horizon_col].values, preds, h_name)
                model_metrics.update(metrics)

                all_y_true.extend(y_test[horizon_col].values)
                all_y_pred.extend(preds)

            overall_mae = mean_absolute_error(all_y_true, all_y_pred)
            overall_rmse = np.sqrt(mean_squared_error(all_y_true, all_y_pred))
            overall_r2 = r2_score(all_y_true, all_y_pred)

            model_metrics["overall_mae"] = float(overall_mae)
            model_metrics["overall_rmse"] = float(overall_rmse)
            model_metrics["overall_r2"] = float(overall_r2)

            for k, v in model_metrics.items():
                mlflow.log_metric(k, round(v, 3))

            results[model_name] = model_metrics
            fitted_models[model_name] = horizon_models

            print(f"{model_name} Results -> Overall RMSE: {overall_rmse:.2f} | MAE: {overall_mae:.2f} | R²: {overall_r2:.3f}")
            for h in ["24h", "48h", "72h"]:
                print(f"     • Day {h[:2]} ({h}): RMSE = {model_metrics[f'{h}_rmse']:.2f} | MAE = {model_metrics[f'{h}_mae']:.2f} | R² = {model_metrics[f'{h}_r2']:.3f}")

    results_df = pd.DataFrame(results).T.sort_values("overall_rmse")
    print("MODEL COMPARISON LEADERBOARD (Sorted by RMSE)")
    print(results_df[["overall_rmse", "overall_mae", "overall_r2", "24h_rmse", "48h_rmse", "72h_rmse"]].round(2))

    best_model_name = results_df.index[0]
    print(f"\nWinning Model: [{best_model_name}] with Overall RMSE = {results_df.loc[best_model_name, 'overall_rmse']:.2f}")

    print("\nGenerating SHAP Feature Importance Explanations...")
    generate_shap_plots(fitted_models[best_model_name]["target_aqi_24h"], X_train, X_test, feature_cols)

    save_and_register_best_model(
        best_model_name=best_model_name,
        best_models_dict=fitted_models[best_model_name],
        feature_cols=feature_cols,
        results_df=results_df,
    )

    return results_df


def generate_shap_plots(model_24h, X_train, X_test, feature_cols):
    sample_test = X_test.iloc[:200]
    est = model_24h.named_steps["regressor"] if hasattr(model_24h, "named_steps") else model_24h

    if isinstance(est, (lgb.LGBMRegressor, RandomForestRegressor)):
        explainer = shap.TreeExplainer(est)
        shap_values = explainer.shap_values(sample_test)
    else:
        explainer = shap.Explainer(est.predict, shap.sample(X_train, 100))
        shap_values = explainer(sample_test).values

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, sample_test, feature_names=feature_cols, max_display=15, show=False)  # type: ignore

    plt.title("SHAP Feature Importance Summary (Top 15 Drivers of AQI)", fontsize=12)
    plt.tight_layout()

    plot_path = PLOTS_DIR / "shap_summary.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"SHAP plot saved to: {plot_path}")


def save_and_register_best_model(best_model_name, best_models_dict, feature_cols, results_df):
    model_bundle = {
        "model_name": best_model_name,
        "models": best_models_dict,
        "feature_cols": feature_cols,
        "metrics": results_df.loc[best_model_name].to_dict(),
        "forecast_horizons": [24, 48, 72],
    }

    local_model_path = MODELS_DIR / "best_model.pkl"
    joblib.dump(model_bundle, local_model_path)
    print(f"Saved local model to: {local_model_path}")

    with mlflow.start_run(run_name=f"REGISTERED_{best_model_name}"):
        mlflow.log_params({"best_model": best_model_name})
        for k, v in model_bundle["metrics"].items():
            mlflow.log_metric(f"final_{k}", v)

        if (PLOTS_DIR / "shap_summary.png").exists():
            mlflow.log_artifact(str(PLOTS_DIR / "shap_summary.png"), artifact_path="plots")
        mlflow.log_artifact(str(local_model_path), artifact_path="model")

        mlflow.sklearn.log_model(
            sk_model=model_bundle["models"]["target_aqi_24h"],
            artifact_path="lgbm_model",
            serialization_format="cloudpickle",
            registered_model_name=MODEL_REGISTRY_NAME,
        )
        print(f"Registered [{MODEL_REGISTRY_NAME}] to DagsHub MLflow Model Registry")


if __name__ == "__main__":
    train_and_evaluate_all()
