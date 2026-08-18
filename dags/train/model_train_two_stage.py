"""
Phase 3 Model Training DAG: Two-Stage Ensemble with Advanced Features
════════════════════════════════════════════════════════════════════════════════

Combines:
  ✓ Phase 3a: Advanced Feature Engineering (19 new features)
  ✓ Phase 3b: Two-Stage Ensemble Model (binary churn + timing prediction)

Expected Results:
  ✓ Class 0 Recall: 64% → 82-85%
  ✓ Class 1 Recall: 66% → 83-85%
  ✓ Overall F1: 0.7928 → 0.8650+

Architecture:
  Stage 1: Binary churn detector (Churn vs No-Churn) → 85%+ recall
  Stage 2: Timing predictor (Early vs Medium churn) → 75%+ accuracy

  Flow:
    Input → Stage 1 → No-Churn? → Output Class 2
                   → Churn? → Stage 2 → Early? → Output Class 0
                                     → Medium? → Output Class 1
"""

from src.les.train.train import ModelTraing
from src.les.train.feature_engineering_v2 import create_advanced_features
from src.les.train.two_stage_model import TwoStageEnsembleModel
from airflow import DAG
from airflow.sdk import task
from datetime import datetime
from pathlib import Path

import pandas as pd
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# ================================
# PATHS
# ================================
PREPROCESSOR_PATH = "/usr/local/airflow/include/artifacts/preprocessor.pkl"
ARTIFACT_DIR = Path("/usr/local/airflow/include/artifacts")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path("/usr/local/airflow/include/data")

BALANCED_DATA_PATH = DATA_DIR / "df_merged_balanced.parquet"


# ================================
# TRANSFORM + TRAIN WITH TWO-STAGE MODEL
# ================================
def transform_train_two_stage():
    try:
        # ✅ Load balanced merged parquet
        df = pd.read_parquet(BALANCED_DATA_PATH)

        print(f"📊 Loaded balanced data: {len(df)} rows")
        print(f"📊 Class distribution:\n{df['Churned'].value_counts().sort_index()}")

        # ✅ PHASE 3a: CREATE ADVANCED FEATURES
        print(f"\n🚀 PHASE 3a: Creating advanced features...")
        df = create_advanced_features(df)
        print(f"✅ Advanced features created: {len(df.columns)} total columns")

        TARGET = "Churned"
        X = df.drop(TARGET, axis=1)

        le = LabelEncoder()
        y = le.fit_transform(df[TARGET])

        # ===================================
        # CHECK FOR DATA LEAKAGE
        # ===================================
        print(f"\n🔍 LEAKAGE CHECK:")
        print(f"   Features shape: {X.shape}")
        print(f"   Target shape: {y.shape}")
        print(f"   Target values: {set(y)}")

        if TARGET in X.columns:
            print(f"   ❌ WARNING: Target '{TARGET}' found in features!")
        if 'Churned' in X.columns:
            print(f"   ❌ WARNING: 'Churned' column found in features!")

        # ===================================
        # DROP PROBLEMATIC COLUMNS
        # ===================================
        leakage_cols = ['DaysToContractEnd', 'end_date']
        pause_cols = ['WeeksPausedToDate', 'PauseCount', 'TotalDaysPaused', 'CurrentlyPaused',
                      'DaysSinceLastPauseEnded', 'LongestPauseDuration', 'AvgPauseDuration']
        covid_cols = ['COVID19_PauseCount']
        weak_cols = ['Discount', 'DaysSinceLastVisit']
        cols_to_drop = leakage_cols + pause_cols + covid_cols + weak_cols

        print(f"🗑️  Dropping leakage: {leakage_cols}")
        print(f"🗑️  Dropping pause: {pause_cols}")
        print(f"🗑️  Dropping COVID: {covid_cols}")
        print(f"🗑️  Dropping weak: {weak_cols}")

        X = X.drop(columns=[col for col in cols_to_drop if col in X.columns])

        # Drop IDs
        for col in ["MembershipID", "member_id"]:
            if col in X.columns:
                X = X.drop(col, axis=1)

        # ===================================
        # CREATE PREPROCESSOR - AUTO-DETECT TYPES
        # ===================================
        numeric_features = X.select_dtypes(include=['int64', 'int32', 'float64', 'float32']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'string', 'category']).columns.tolist()

        print(f"📊 Numeric features ({len(numeric_features)}): {numeric_features[:5]}... (showing first 5)")
        print(f"📊 Categorical features ({len(categorical_features)}): {categorical_features[:5]}... (showing first 5)")
        print(f"🔍 Missing values in numeric: {X[numeric_features].isnull().sum().sum()}")
        print(f"🔍 Missing values in categorical: {X[categorical_features].isnull().sum().sum()}")

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]), numeric_features),
                ('cat', Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
                ]), categorical_features)
            ],
            remainder='drop'
        )

        # Split
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print(f"\n✅ Data prepared:")
        print(f"   Train size: {len(x_train)}, Test size: {len(x_test)}")

        # ✅ PHASE 3b: TRAIN TWO-STAGE ENSEMBLE MODEL
        print(f"\n🚀 PHASE 3b: Training two-stage ensemble model...")
        two_stage_model = TwoStageEnsembleModel()
        results = two_stage_model.train(
            x_train, y_train, x_test, y_test, preprocessor
        )

        # ===================================
        # SAVE ARTIFACTS
        # ===================================
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save models
        model_paths = two_stage_model.save_models(ARTIFACT_DIR)

        # Save combined results
        combined_results = results['combined']
        metrics = {
            "accuracy": float(combined_results['accuracy']),
            "balanced_accuracy": float(combined_results['balanced_accuracy']),
            "f1_weighted": float(combined_results['f1_weighted']),
            "rows": len(df),
            "timestamp": timestamp,
            "model_type": "two_stage_ensemble",
            "stage1_path": model_paths['stage1_path'],
            "stage2_path": model_paths['stage2_path'],
            "class_distribution": df['Churned'].value_counts().sort_index().to_dict()
        }

        metrics_path = ARTIFACT_DIR / f"metrics_two_stage_{timestamp}.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)

        # Save log with detailed results
        log_path = ARTIFACT_DIR / f"log_two_stage_{timestamp}.txt"
        with open(log_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("MODEL TRAINING REPORT - TWO-STAGE ENSEMBLE\n")
            f.write("=" * 80 + "\n\n")

            f.write("PHASE 3: TWO-STAGE ENSEMBLE WITH ADVANCED FEATURES\n")
            f.write("-" * 80 + "\n")
            f.write("Stage 1: Binary churn detector (Churn vs No-Churn)\n")
            f.write("Stage 2: Timing predictor (Early vs Medium churn)\n")
            f.write("Features: 19 new discriminative features added\n\n")

            f.write("OVERALL RESULTS:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Accuracy: {combined_results['accuracy']:.4f}\n")
            f.write(f"Balanced Accuracy: {combined_results['balanced_accuracy']:.4f}\n")
            f.write(f"F1-Weighted: {combined_results['f1_weighted']:.4f}\n\n")

            f.write("CONFUSION MATRIX:\n")
            f.write("-" * 80 + "\n")
            cm = combined_results['confusion_matrix']
            f.write("Rows: True Labels | Columns: Predicted Labels\n")
            f.write("       Class 0  Class 1  Class 2\n")
            for i, row in enumerate(cm):
                f.write(f"Class {i}:  {row[0]:6d}  {row[1]:6d}  {row[2]:6d}\n")
            f.write("\n")

            f.write("CLASSIFICATION REPORT:\n")
            f.write("-" * 80 + "\n")
            f.write(combined_results['report'])
            f.write("\n")

            f.write("STAGE 1 (Binary Churn Detection) RESULTS:\n")
            f.write("-" * 80 + "\n")
            if 'stage1' in results:
                stage1 = results['stage1']
                f.write(f"Recall: {stage1['recall']:.4f}\n")
                f.write(f"Precision: {stage1['precision']:.4f}\n")
                f.write(f"F1-Score: {stage1['f1']:.4f}\n\n")

            f.write("STAGE 2 (Timing Prediction) RESULTS:\n")
            f.write("-" * 80 + "\n")
            if results.get('stage2'):
                stage2 = results['stage2']
                f.write(f"Recall: {stage2['recall']:.4f}\n")
                f.write(f"Precision: {stage2['precision']:.4f}\n")
                f.write(f"F1-Score: {stage2['f1']:.4f}\n\n")

        print("📦 Artifacts saved (Two-Stage):")
        print(model_paths['stage1_path'])
        print(str(metrics_path))
        print(str(log_path))

        return {
            "stage1_path": model_paths['stage1_path'],
            "stage2_path": model_paths['stage2_path'],
            "metrics_path": str(metrics_path),
            "log_path": str(log_path),
            "accuracy": float(combined_results['accuracy']),
            "model_type": "two_stage_ensemble"
        }

    except Exception as e:
        raise Exception(f"Two-stage model training failed: {e}")


# ================================
# DAG
# ================================
with DAG(
    dag_id="ModelTrainTwoStage",
    description="Les Mills two-stage churn model with advanced features (Phase 3)",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False
) as dag:

    @task
    def train_task():
        return transform_train_two_stage()

    train_task()
