import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

DATA_DIR = "data"
OUTPUT_DIR = "output"


def calculate_mape(y_true, y_pred):
    """Menghitung Mean Absolute Percentage Error (%)."""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def evaluate_model(model, X_test, y_test, is_log_target=True):
    """Menghitung metrik performa model pada data uji."""
    preds = model.predict(X_test)
    if is_log_target:
        preds = np.expm1(preds)
    return {
        "R2 Score": round(r2_score(y_test, preds), 4),
        "MAE (Juta Rp)": round(mean_absolute_error(y_test, preds) / 1_000_000, 2),
        "RMSE (Juta Rp)": round(np.sqrt(mean_squared_error(y_test, preds)) / 1_000_000, 2),
        "MAPE (%)": round(calculate_mape(y_test, preds), 2)
    }


def run_experiments():
    """Menjalankan ablation study dan hyperparameter tuning untuk Random Forest dan XGBoost."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df_tanpa = pd.read_csv(os.path.join(DATA_DIR, "dataset_tanpa_fe.csv"))
    df_dengan = pd.read_csv(os.path.join(DATA_DIR, "dataset_dengan_fe.csv"))

    y = df_dengan["harga"]
    y_log = np.log1p(y)

    num_cols_base = [
        "luas_tanah", "luas_bangunan", "kamar_tidur", "kamar_mandi",
        "lantai", "carport", "keamanan", "taman", "latitude", "longitude"
    ]
    num_cols_fe = num_cols_base + ["jarak_ke_pusat_kota", "rasio_bangunan_tanah", "total_ruangan"]

    prep_tanpa = ColumnTransformer([
        ("num", "passthrough", num_cols_base),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["furnished"])
    ])
    prep_dengan = ColumnTransformer([
        ("num", "passthrough", num_cols_fe),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["furnished", "kecamatan"])
    ])

    idx_train, idx_test = train_test_split(np.arange(len(df_dengan)), test_size=0.20, random_state=42)
    y_train, y_test = y.iloc[idx_train], y.iloc[idx_test]
    y_log_train = y_log.iloc[idx_train]

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    best_params = {}

    # Ruang hyperparameter
    rf_dist = {
        "rf__n_estimators": [100, 150, 200, 250],
        "rf__max_depth": [10, 15, 20, None],
        "rf__min_samples_split": [2, 5, 10],
        "rf__min_samples_leaf": [1, 2, 4]
    }
    xgb_dist = {
        "xgb__n_estimators": [150, 200, 250, 300],
        "xgb__max_depth": [4, 6, 8],
        "xgb__learning_rate": [0.02, 0.05, 0.1],
        "xgb__subsample": [0.8, 0.9, 1.0],
        "xgb__colsample_bytree": [0.8, 1.0]
    }

    scenarios = [
        ("Random Forest", "Tanpa FE (Baseline)", prep_tanpa, df_tanpa, "rf",
         RandomForestRegressor(random_state=42, n_jobs=-1), rf_dist),
        ("Random Forest", "DENGAN Feature Engineering", prep_dengan, df_dengan, "rf",
         RandomForestRegressor(random_state=42, n_jobs=-1), rf_dist),
        ("XGBoost", "Tanpa FE (Baseline)", prep_tanpa, df_tanpa, "xgb",
         XGBRegressor(random_state=42, n_jobs=-1), xgb_dist),
        ("XGBoost", "DENGAN Feature Engineering", prep_dengan, df_dengan, "xgb",
         XGBRegressor(random_state=42, n_jobs=-1), xgb_dist),
    ]

    best_xgb_model = None

    for model_name, scen_name, prep, df_src, prefix, estimator, dist in scenarios:
        pipe = Pipeline([("prep", prep), (prefix, estimator)])

        # Coarse search
        rand = RandomizedSearchCV(pipe, dist, n_iter=8, cv=cv, scoring="neg_mean_squared_error",
                                  random_state=42, n_jobs=-1)
        rand.fit(df_src.iloc[idx_train], y_log_train)
        bp = rand.best_params_

        # Fine tuning
        grid_params = {k: [v] for k, v in bp.items()}
        if f"{prefix}__min_samples_split" in bp:
            grid_params[f"{prefix}__min_samples_split"] = [bp[f"{prefix}__min_samples_split"],
                                                           max(2, bp[f"{prefix}__min_samples_split"] - 1)]

        grid = GridSearchCV(pipe, grid_params, cv=cv, scoring="neg_mean_squared_error", n_jobs=-1)
        grid.fit(df_src.iloc[idx_train], y_log_train)

        fitted_model = grid.best_estimator_
        key_name = f"{model_name}_{scen_name}".replace(" ", "_")
        best_params[key_name] = {k.replace(f"{prefix}__", ""): v for k, v in grid.best_params_.items()}

        metrics = evaluate_model(fitted_model, df_src.iloc[idx_test], y_test, is_log_target=True)
        results.append({"Model": model_name, "Skenario": scen_name, **metrics})

        if model_name == "XGBoost" and scen_name == "DENGAN Feature Engineering":
            best_xgb_model = fitted_model

    # Export artefak model
    if best_xgb_model:
        joblib.dump(best_xgb_model, os.path.join(OUTPUT_DIR, "model_terbaik.joblib"))

        # Ekstraksi feature importance
        prep_step = best_xgb_model.named_steps["prep"]
        xgb_step = best_xgb_model.named_steps["xgb"]
        cat_features = list(prep_step.named_transformers_["cat"].get_feature_names_out(["furnished", "kecamatan"]))
        all_features = num_cols_fe + cat_features

        fi_df = pd.DataFrame({
            "Fitur": all_features,
            "Importance": xgb_step.feature_importances_
        }).sort_values(by="Importance", ascending=False)
        fi_df.to_csv(os.path.join(OUTPUT_DIR, "feature_importance_xgboost.csv"), index=False)

    with open(os.path.join(OUTPUT_DIR, "best_hyperparameters.json"), "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)

    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(OUTPUT_DIR, "tabel_hasil_eksperimen.csv"), index=False)
    print(df_results.to_string(index=False))
    return df_results


if __name__ == "__main__":
    run_experiments()
