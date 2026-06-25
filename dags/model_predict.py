from src.les.train.run import snapshot as snapshot_fn
from src.les.train.readdata import read_data as read_data_fn
from src.les.train.ingestfortrain import ingest_data
from airflow import DAG
from airflow.sdk import task
from airflow.models import Variable
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import joblib

# ================================
# PATHS
# ================================
DATA_DIR = Path("/usr/local/airflow/include/data")
ARTIFACT_DIR = Path("/usr/local/airflow/include/artifacts")
DATA_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

# ================================
# PREDICT DATE (from Airflow Variable)
# ================================
PREDICT_DATE = "2024-01-01"


# ================================
# APPLY MODEL & PREDICT
# ================================
def apply_model(data_paths: dict):
    """
    Load trained model and make predictions.
    Input: data_paths with 'df_merged_path' key
    Returns: predictions DataFrame with 3 classes (1, 2, 3)
    """
    try:
        # Load features (already has ews_pct, risk_band merged)
        df = pd.read_parquet(data_paths["df_merged_path"])

        # Drop Churned if present (we're predicting, not training)
        if 'Churned' in df.columns:
            df = df.drop('Churned', axis=1)

        print(f"✅ Loaded features: {len(df)} rows × {len(df.columns)} cols")

        # Find latest model
        model_files = sorted(ARTIFACT_DIR.glob("model_balanced_*.pkl"), reverse=True)
        if not model_files:
            raise FileNotFoundError("❌ No trained model found!")

        model_path = model_files[0]
        print(f"📦 Model: {model_path.name}")
        model = joblib.load(model_path)

        # Feature order (must match training)
        model_features = [
            'Gender', 'Age', 'SubCategory', 'Category', 'Channel', 'ClubName',
            'MembershipTypeDesc', 'RegularPayment', 'Amount', 'Base Amount',
            'PaymentFrequency', 'Term', 'TermDays', 'BillingDay', 'Lump Sum Flag',
            'DaysSinceLastAccessed', 'TenureDays', 'DaysSinceOriginalStart',
            'TotalAttendanceToDate', 'Visits_Last30d', 'Visits_Last90d',
            'Transferred_IN', 'Transferred_OUT', 'EndedWithinCoolingPeriod',
            'ews_pct', 'risk_band'
        ]

        # Ensure all features exist
        for col in model_features:
            if col not in df.columns:
                print(f"⚠️  Missing: {col}, filling with 0")
                df[col] = 0

        # DO NOT fill missing values! Let the model's preprocessor handle it
        # The preprocessor was trained to fill:
        # - Numeric: median
        # - Categorical: most_frequent
        X = df[model_features]

        print(f"🔍 Feature check:")
        print(f"   Missing values: {X.isnull().sum().sum()}")
        print(f"   Features shape: {X.shape}")

        # Multi-class prediction (classes 1, 2, 3)
        y_pred_proba = model.predict_proba(X)  # Shape: (n_samples, 3 or 2)
        y_pred = model.predict(X)

        print(f"🔍 Model prediction debug:")
        print(f"   Unique predictions: {np.unique(y_pred)}")
        print(f"   Prediction shape: {y_pred.shape}")
        print(f"   Proba shape: {y_pred_proba.shape}")
        print(f"   Sample predictions: {y_pred[:10]}")

        # Use model predictions directly (0, 1, 2) - no mapping
        # 0 = Churn 0-3 months
        # 1 = Churn 3-6 months
        # 2 = All others (no churn OR churn after 6 months)
        y_pred_labeled = y_pred

        print(f"✅ Using model predictions directly (0, 1, 2)")
        print(f"   Class 0: Churn 0-3 months")
        print(f"   Class 1: Churn 3-6 months")
        print(f"   Class 2: All others")

        # Create results with all 3 class probabilities
        class_label_map = {0: 'Churn_0-3mo', 1: 'Churn_3-6mo', 2: 'NoChurn'}

        # Calculate confidence (max probability across all classes)
        confidence = y_pred_proba.max(axis=1)

        predictions = pd.DataFrame({
            'MembershipID': df['MembershipID'],
            'Class0_Prob': y_pred_proba[:, 0],  # Churn (0-3 mo)
            'Class1_Prob': y_pred_proba[:, 1],  # Churn (3-6 mo)
            'Class2_Prob': y_pred_proba[:, 2],  # All others
            'PredictedClass': y_pred_labeled,
            'Confidence': confidence,  # Max probability (model confidence)
            'ClassLabel': [class_label_map[c] for c in y_pred_labeled]
        })

        print(f"✅ Predictions (3-class):")
        print(f"   Total: {len(predictions)}")
        print(f"   Class 0 (Churn 0-3 months): {(y_pred_labeled == 0).sum()} ({(y_pred_labeled == 0).sum()/len(predictions)*100:.1f}%)")
        print(f"   Class 1 (Churn 3-6 months): {(y_pred_labeled == 1).sum()} ({(y_pred_labeled == 1).sum()/len(predictions)*100:.1f}%)")
        print(f"   Class 2 (All others): {(y_pred_labeled == 2).sum()} ({(y_pred_labeled == 2).sum()/len(predictions)*100:.1f}%)")

        return {
            "predictions_df": predictions,
            "predictions_count": len(predictions)
        }

    except Exception as e:
        raise Exception(f"Model prediction failed: {e}")


# ================================
# SAVE PREDICTIONS WITH ALL FEATURES
# ================================
def save_predictions(predictions_and_data: dict = None):
    """
    Save predictions to CSV with all original features.
    Includes ChurnedPrediction column (1, 2, 3 for class labels).
    Can run independently by loading latest predictions from disk.
    """
    try:
        # If no data passed, try to load from latest computation
        if predictions_and_data is None or "features_path" not in predictions_and_data:
            print("⚠️  Running save_predictions independently...")
            latest_parquet = max(
                DATA_DIR.glob("df_merged_*.parquet"),
                default=None,
                key=lambda p: p.stat().st_mtime
            )
            if not latest_parquet:
                raise Exception("No feature data found. Run full DAG first.")

            features_path = str(latest_parquet)
            print(f"   Using features: {latest_parquet.name}")

            # Reconstruct predictions from features
            features = pd.read_parquet(features_path)
            # Use predictions from CSV if available
            latest_pred_csv = max(
                ARTIFACT_DIR.glob("predictions_*.csv"),
                default=None,
                key=lambda p: p.stat().st_mtime
            )
            if latest_pred_csv:
                pred_df = pd.read_csv(latest_pred_csv)
                churned_labels = pred_df['ChurnedPrediction'].values if 'ChurnedPrediction' in pred_df.columns else None
            else:
                raise Exception("No predictions found.")
        else:
            # Get predictions and merge with original features
            predictions_df = predictions_and_data["predictions_df"]
            features_path = predictions_and_data["features_path"]

            # Load original features
            features = pd.read_parquet(features_path)

            print(f"🔍 Debug - Features loaded:")
            print(f"   Columns: {list(features.columns)}")
            print(f"   Has 'end_date': {'end_date' in features.columns}")
            print(f"   Shape: {features.shape}")

            # Use predicted classes directly (1, 2, 3)
            churned_labels = predictions_df['PredictedClass'].values

        # Merge features with predictions
        result_df = features.copy()
        result_df['ChurnedPrediction'] = churned_labels

        # Add RunDate column (from predict_date)
        run_date = Variable.get("run_date", default_var="")
        result_df['RunDate'] = run_date

        # Ensure end_date is returned as date format and rename to EndDate
        if 'end_date' in result_df.columns:
            # Convert to datetime if not already
            result_df['end_date'] = pd.to_datetime(result_df['end_date'], errors='coerce')
            # Rename to EndDate (capitalized)
            result_df = result_df.rename(columns={'end_date': 'EndDate'})
            print(f"✅ EndDate formatted as date")
        else:
            print(f"⚠️  EndDate not found")

        # Reorganize columns: key validation columns first, then others
        key_cols = ['MembershipID', 'ChurnedPrediction']
        if 'EndDate' in result_df.columns:
            key_cols.append('EndDate')  # Add EndDate for validation
        key_cols.extend(['RunDate'])  # Add RunDate after

        # Add remaining columns
        other_cols = [col for col in result_df.columns if col not in key_cols]
        column_order = key_cols + other_cols

        result_df = result_df[column_order]

        # Save to CSV
        csv_path = ARTIFACT_DIR / f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        result_df.to_csv(csv_path, index=False)

        print(f"✅ Predictions saved to CSV:")
        print(f"   Path: {csv_path}")
        print(f"   Rows: {len(result_df)}")
        print(f"   Cols: {len(result_df.columns)}")
        print(f"\n📋 Column Order:")
        print(f"   Key validation: MembershipID | ChurnedPrediction | EndDate | RunDate")
        print(f"   + {len(other_cols)} feature columns")
        print(f"\n📊 Class Distribution:")
        print(f"   Class 0 (Churn 0-3 months): {(churned_labels == 0).sum()} ({(churned_labels == 0).sum()/len(result_df)*100:.1f}%)")
        print(f"   Class 1 (Churn 3-6 months): {(churned_labels == 1).sum()} ({(churned_labels == 1).sum()/len(result_df)*100:.1f}%)")
        print(f"   Class 2 (All others): {(churned_labels == 2).sum()} ({(churned_labels == 2).sum()/len(result_df)*100:.1f}%)")

        return {"predictions_csv_path": str(csv_path)}

    except Exception as e:
        raise Exception(f"Saving predictions failed: {e}")


# ================================
# DAG
# ================================
with DAG(
    dag_id="ModelPredict",
    description="Prepare data + apply trained model for 3-class prediction (1, 2, 3)",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False
) as dag:

    @task
    def read_data_task():
        predict_date = Variable.get("predict_date", default_var=PREDICT_DATE)
        Variable.set("train_date", predict_date)
        Variable.set("run_date", predict_date)
        print(f"📅 Predicting for: {predict_date}")
        return read_data_fn()

    @task
    def snapshots_task(df):
        return snapshot_fn(df)

    @task
    def ingest_task(snap_df):
        # Call ingest_data - it ALREADY merges snapshots internally!
        data_paths = ingest_data(snap_df)

        # Load features (snapshots already merged by ingest_data)
        features_df = pd.read_parquet(data_paths["df_merged_path"])

        # Remove Churned column (we're predicting, not training)
        if 'Churned' in features_df.columns:
            features_df = features_df.drop('Churned', axis=1)

        # Verify snapshot columns are present
        print(f"✅ Snapshot merge verified:")
        print(f"   Has ews_pct: {'ews_pct' in features_df.columns}")
        print(f"   Has risk_band: {'risk_band' in features_df.columns}")
        print(f"   Has end_date: {'end_date' in features_df.columns}")

        features_df.to_parquet(data_paths["df_merged_path"])

        print(f"✅ Features ready: {len(features_df)} rows × {len(features_df.columns)} cols")

        return data_paths

    @task
    def model_task(data_paths):
        predictions_df = apply_model(data_paths)
        return {
            "predictions_df": predictions_df["predictions_df"],
            "features_path": data_paths["df_merged_path"],
            "predictions_count": predictions_df["predictions_count"]
        }

    @task
    def save_task(predictions_data=None):
        # Allow running task independently (without model_task output)
        if predictions_data is None:
            print("⏭️  save_task running independently...")
        return save_predictions(predictions_data)

    # Flow
    raw_df = read_data_task()
    snap_df = snapshots_task(raw_df)
    data_paths = ingest_task(snap_df)
    model_output = model_task(data_paths)
    save_task(model_output)
