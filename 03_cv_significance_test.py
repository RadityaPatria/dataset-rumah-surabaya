import json
import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

DATA_DIR = "data"
OUTPUT_DIR = "output"


def run_cv_significance_test():
    """Menjalankan Repeated Cross-Validation dan paired t-test untuk menguji signifikansi Feature Engineering."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df_tanpa = pd.read_csv(os.path.join(DATA_DIR, "dataset_tanpa_fe.csv"))
    df_dengan = pd.read_csv(os.path.join(DATA_DIR, "dataset_dengan_fe.csv"))

    y = df_dengan["harga"]
    y_log = np.log1p(y)

    with open(os.path.join(OUTPUT_DIR, "best_hyperparameters.json"), "r", encoding="utf-8") as f:
        best_params = json.load(f)

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

    rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)
    n_folds = rkf.get_n_splits()
    print(f"Menjalankan Repeated K-Fold Cross Validation (5-Fold, 10 Repeats = {n_folds} Folds)...")

    # Inisialisasi model dari hyperparameter terbaik
    xgb_tanpa_params = best_params.get("XGBoost_Tanpa_FE_(Baseline)") or best_params.get("XGBoost_Tanpa_FE", {})
    xgb_dengan_params = best_params.get("XGBoost_DENGAN_Feature_Engineering") or best_params.get("XGBoost_Dengan_FE", {})
    rf_tanpa_params = best_params.get("Random_Forest_Tanpa_FE_(Baseline)") or best_params.get("Random_Forest_Tanpa_FE", {})
    rf_dengan_params = best_params.get("Random_Forest_DENGAN_Feature_Engineering") or best_params.get("Random_Forest_Dengan_FE", {})

    models = {
        "XGBoost": {
            "pipe_tanpa": Pipeline([
                ("prep", prep_tanpa),
                ("xgb", XGBRegressor(random_state=42, n_jobs=-1, **xgb_tanpa_params))
            ]),
            "pipe_dengan": Pipeline([
                ("prep", prep_dengan),
                ("xgb", XGBRegressor(random_state=42, n_jobs=-1, **xgb_dengan_params))
            ]),
            "scores_tanpa": [],
            "scores_dengan": []
        },
        "Random Forest": {
            "pipe_tanpa": Pipeline([
                ("prep", prep_tanpa),
                ("rf", RandomForestRegressor(random_state=42, n_jobs=-1, **rf_tanpa_params))
            ]),
            "pipe_dengan": Pipeline([
                ("prep", prep_dengan),
                ("rf", RandomForestRegressor(random_state=42, n_jobs=-1, **rf_dengan_params))
            ]),
            "scores_tanpa": [],
            "scores_dengan": []
        }
    }

    # Evaluasi pada tiap fold
    fold_idx = 1
    for train_idx, test_idx in rkf.split(df_dengan):
        y_train_log = y_log.iloc[train_idx]
        y_test_true = y.iloc[test_idx]

        for model_name, m_dict in models.items():
            # 1. Tanpa Feature Engineering
            m_dict["pipe_tanpa"].fit(df_tanpa.iloc[train_idx], y_train_log)
            pred_log_tanpa = m_dict["pipe_tanpa"].predict(df_tanpa.iloc[test_idx])
            r2_tanpa = r2_score(y_test_true, np.expm1(pred_log_tanpa))
            m_dict["scores_tanpa"].append(r2_tanpa)

            # 2. DENGAN Feature Engineering
            m_dict["pipe_dengan"].fit(df_dengan.iloc[train_idx], y_train_log)
            pred_log_dengan = m_dict["pipe_dengan"].predict(df_dengan.iloc[test_idx])
            r2_dengan = r2_score(y_test_true, np.expm1(pred_log_dengan))
            m_dict["scores_dengan"].append(r2_dengan)

        if fold_idx % 10 == 0 or fold_idx == n_folds:
            print(f"  Progress: {fold_idx}/{n_folds} folds selesai...")
        fold_idx += 1

    summary_rows = []
    print("\n" + "=" * 90)
    print("HASIL REPEATED CROSS-VALIDATION (5-Fold x 10 Repeats = 50 Folds)")
    print("=" * 90)

    for model_name, m_dict in models.items():
        arr_tanpa = np.array(m_dict["scores_tanpa"])
        arr_dengan = np.array(m_dict["scores_dengan"])

        mean_tanpa, std_tanpa = np.mean(arr_tanpa), np.std(arr_tanpa)
        mean_dengan, std_dengan = np.mean(arr_dengan), np.std(arr_dengan)

        # Paired t-test
        t_stat, p_val = stats.ttest_rel(arr_dengan, arr_tanpa)
        is_sig = p_val < 0.05
        kesimpulan = "Signifikan (p < 0.05)" if is_sig else "Tidak Signifikan (p >= 0.05)"

        summary_rows.append({
            "Model": model_name,
            "R2_Tanpa_FE_Mean": round(mean_tanpa, 4),
            "R2_Tanpa_FE_Std": round(std_tanpa, 4),
            "R2_Dengan_FE_Mean": round(mean_dengan, 4),
            "R2_Dengan_FE_Std": round(std_dengan, 4),
            "T_Statistic": round(t_stat, 4),
            "P_Value": f"{p_val:.4e}" if p_val < 0.0001 else f"{p_val:.4f}",
            "Signifikan_0_05": kesimpulan
        })

        print(f"[{model_name}]")
        print(f"  - Tanpa FE : R² = {mean_tanpa:.4f} ± {std_tanpa:.4f}")
        print(f"  - Dengan FE: R² = {mean_dengan:.4f} ± {std_dengan:.4f}")
        print(f"  - Paired t-test: t = {t_stat:.4f}, p-value = {p_val:.4e} -> {kesimpulan}")
        print("-" * 90)

    df_summary = pd.DataFrame(summary_rows)
    out_csv = os.path.join(OUTPUT_DIR, "cv_significance_results.csv")
    df_summary.to_csv(out_csv, index=False)
    print(f"\nHasil berhasil disimpan ke: {out_csv}")
    return df_summary


if __name__ == "__main__":
    run_cv_significance_test()
