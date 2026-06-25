"""
Enhanced training DAG using comprehensive feature set (36 features).
Uses ingestfortrain_enhanced.py for richer feature engineering.
"""

from src.les.train.run import snapshot as snapshot_fn
from src.les.train.readdata import read_data as read_data_fn
from src.les.train.ingestfortrain import ingest_data
from src.les.train.train import ModelTraing

from airflow import DAG
from airflow.sdk import task
from datetime import datetime
from pathlib import Path

import pandas as pd
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

# ================================
# PATHS
# ================================
PREPROCESSOR_PATH = "/usr/local/airflow/include/artifacts/preprocessor.pkl"
ARTIFACT_DIR = Path("/usr/local/airflow/include/artifacts")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


# ================================
# TRANSFORM + TRAIN WITH ENHANCED FEATURES
# ================================
def transform_and_train_enhanced(data_paths: dict):
    try:
        df = pd.read_parquet(data_paths["df_merged_path"])

        logger_info = []
        logger_info.append(f"✅ Loaded enhanced merged data: {len(df)} rows")
        logger_info.append(f"📊 Features available: {len(df.columns)}")
        logger_info.append(f"📋 Columns: {list(df.columns)}")

        TARGET = "Churned"
        X = df.drop(TARGET, axis=1)

        le = LabelEncoder()
        y = le.fit_transform(df[TARGET])

        # Drop IDs
        for col in ["MembershipID", "member_id"]:
            if col in X.columns:
                X = X.drop(col, axis=1)

        logger_info.append(f"📊 Features for training: {len(X.columns)}")
        logger_info.append(f"🎯 Target distribution:\n{pd.Series(y).value_counts().sort_index()}")

        # Split
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        logger_info.append(f"📈 Train size: {len(x_train)}, Test size: {len(x_test)}")

        # Load preprocessor
        preprocessor = joblib.load(PREPROCESSOR_PATH)

        # Train model
        model_obj = ModelTraing()
        trained_model, best_model_name, results = model_obj.trainingModel(
            x_train, y_train, x_test, y_test, preprocessor
        )

        # =========================
        # EVALUATION
        # =========================
        y_pred = trained_model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        logger_info.append(f"\n✅ Model Performance (Enhanced Features):")
        logger_info.append(f"   Accuracy:  {accuracy:.4f}")
        logger_info.append(f"   Precision: {precision:.4f}")
        logger_info.append(f"   Recall:    {recall:.4f}")
        logger_info.append(f"   F1-Score:  {f1:.4f}")

        # =========================
        # SAVE ARTIFACTS
        # =========================
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save model
        model_path = ARTIFACT_DIR / f"model_enhanced_{timestamp}.pkl"
        joblib.dump(trained_model, model_path)

        # Save metrics
        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "rows": len(df),
            "features": len(X.columns),
            "timestamp": timestamp,
            "model_type": "enhanced",
            "feature_count": 36,
            "class_distribution": pd.Series(y).value_counts().sort_index().to_dict()
        }

        metrics_path = ARTIFACT_DIR / f"metrics_enhanced_{timestamp}.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)

        # Save log
        log_path = ARTIFACT_DIR / f"log_enhanced_{timestamp}.txt"
        with open(log_path, "w") as f:
            f.write("\n".join(logger_info))

        logger_info.append(f"\n📦 Artifacts saved:")
        logger_info.append(f"   Model: {model_path}")
        logger_info.append(f"   Metrics: {metrics_path}")
        logger_info.append(f"   Log: {log_path}")

        print("\n".join(logger_info))

        return {
            "model_path": str(model_path),
            "metrics_path": str(metrics_path),
            "log_path": str(log_path),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "model_type": "enhanced",
            "feature_count": 36
        }

    except Exception as e:
        raise Exception(f"Enhanced training failed: {e}")


# ================================
# DAG
# ================================
with DAG(
    dag_id="ModelTrainEnhanced",
    description="Les Mills training with enhanced features (36 features from enriched ingest)",
    schedule="@daily",
    start_date=datetime(2025, 1, 1),
    catchup=False
) as dag:

    # -------------- READ DATA ---------------
    @task
    def read_data_task():
        return read_data_fn()

    # -------------- SNAPSHOT ---------------
    @task
    def snapshots_task(df):
        return snapshot_fn(df)

    # -------------- ENHANCED INGEST --------
    @task
    def ingest_task(snap_df):
        return ingest_data(snap_df)

    # -------------- TRAIN ----------------
    @task
    def train_task(data_paths):
        return transform_and_train_enhanced(data_paths)

    # ================================
    # FLOW
    # ================================
    raw_df = read_data_task()
    snap_df = snapshots_task(raw_df)
    data_paths = ingest_task(snap_df)
    train_task(data_paths)
