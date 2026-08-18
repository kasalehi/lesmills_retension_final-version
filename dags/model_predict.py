from src.les.train.run import snapshot as snapshot_fn
from src.les.train.readdata import read_data as read_data_fn
from src.les.train.ingestfortrain import ingest_data
from src.les.train.threshold_tuning import ThresholdTuner  # ✅ Phase 1 thresholds
from src.les.train.feature_engineer_sql import engineer_sql_features  # ✅ Feature engineering
from airflow import DAG
from airflow.sdk import task
from airflow.models import Variable
from airflow.providers.microsoft.mssql.hooks.mssql import MsSqlHook
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import joblib
import logging

logger = logging.getLogger(__name__)

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

        # ✅ PHASE 1: Apply optimized thresholds
        print(f"\n🎯 PHASE 1: Applying optimized thresholds...")
        try:
            # Find latest thresholds
            threshold_files = sorted(ARTIFACT_DIR.glob("optimal_thresholds_*.pkl"), reverse=True)
            if threshold_files:
                thresholds_path = threshold_files[0]
                thresholds = joblib.load(thresholds_path)
                print(f"✅ Thresholds loaded: {thresholds_path.name}")
                print(f"   Thresholds: {thresholds}")

                # Apply thresholds using ThresholdTuner
                tuner = ThresholdTuner()
                y_pred_labeled = tuner.apply_thresholds_to_predictions(
                    y_prob=y_pred_proba,
                    threshold_config=thresholds
                )
                print(f"✅ Applied optimized thresholds")
            else:
                # Fallback: Use default predictions if no thresholds found
                print(f"⚠️  No optimized thresholds found, using default predictions")
                y_pred_labeled = model.predict(X)
        except Exception as e:
            print(f"⚠️  Error loading thresholds: {e}, using default predictions")
            y_pred_labeled = model.predict(X)

        print(f"🔍 Model prediction debug:")
        print(f"   Unique predictions: {np.unique(y_pred_labeled)}")
        print(f"   Prediction shape: {y_pred_labeled.shape}")
        print(f"   Proba shape: {y_pred_proba.shape}")
        print(f"   Sample predictions: {y_pred_labeled[:10]}")

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
# JOIN PREDICTIONS WITH FEATURES
# ================================
def merge_predictions_with_features(predictions_data: dict):
    """
    Join predictions (MembershipID, Prediction, Confidence) with raw features.

    Input:
      - predictions_data: dict with 'predictions_df' and 'features_path'

    Returns:
      - merged_df: raw features + prediction columns
    """
    try:
        print("🔗 Merging predictions with raw features...")

        # Load predictions
        predictions_df = predictions_data["predictions_df"]
        print(f"✅ Predictions loaded: {len(predictions_df)} rows")
        print(f"   Columns: {list(predictions_df.columns)}")

        # Load raw features
        features_path = predictions_data["features_path"]
        features_df = pd.read_parquet(features_path)
        print(f"✅ Features loaded: {len(features_df)} rows")
        print(f"   Columns: {list(features_df.columns)[:10]}... (showing first 10)")

        # JOIN on MembershipID
        merged_df = features_df.merge(
            predictions_df[['MembershipID', 'PredictedClass', 'Confidence', 'ClassLabel']],
            on='MembershipID',
            how='left'
        )

        print(f"✅ Merge complete:")
        print(f"   Result rows: {len(merged_df)}")
        print(f"   Result cols: {len(merged_df.columns)}")
        print(f"   Rows with predictions: {merged_df['PredictedClass'].notna().sum()}")
        print(f"   Rows missing predictions: {merged_df['PredictedClass'].isna().sum()}")

        return {
            "merged_df": merged_df,
            "merge_count": len(merged_df),
            "predictions_matched": merged_df['PredictedClass'].notna().sum()
        }

    except Exception as e:
        raise Exception(f"Merge predictions with features failed: {e}")


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
    description="Apply Phase 1 model + insert predictions to SQL (with feature engineering)",
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
    def merge_predictions_task(model_output):
        # ✅ NEW: Join predictions with raw features
        return merge_predictions_with_features(model_output)

    @task
    def engineer_features_task(merge_output):
        # ✅ NEW: Engineer features from merged data
        print("🔧 Engineering features from merged data...")

        merged_df = merge_output["merged_df"]

        # Engineer features
        engineered_df = engineer_sql_features(merged_df)

        print(f"✅ Features engineered successfully")
        print(f"   Engineered rows: {len(engineered_df)}")
        print(f"   Total columns: {len(engineered_df.columns)}")

        return {
            "merged_df": engineered_df,
            "engineered_count": len(engineered_df),
            "pred_class_col": "PredictedClass",
            "pred_confidence_col": "Confidence"
        }

    @task
    def insert_sql_task(engineered_data=None):
        """
        Insert predictions and engineered features directly into SQL table.
        All logic defined inside this task function.

        Can run independently:
          - If engineered_data passed: Use it (from DAG pipeline)
          - If None: Load from latest parquet files (standalone mode)
        """
        try:
            # ✅ Get predict_date from Airflow Variable (from UI)
            # RunDate = predict_date directly (no conversion)
            run_date = Variable.get("predict_date", default_var="2024-01-01")

            print(f"\n📅 INSERT_SQL_TASK:")
            print(f"   RunDate (from Airflow Variable predict_date): {run_date}")

            # ===================================
            # LOAD DATA
            # ===================================
            print("🔗 Connecting to SQL Database...")
            hook = MsSqlHook(mssql_conn_id="mssql")

            print("📊 Loading merged and engineered data...")

            # ✅ FALLBACK: If no data passed, load from latest parquet (standalone mode)
            if engineered_data is None or "merged_df" not in engineered_data:
                print("⚠️  No engineered_data from pipeline, trying standalone mode...")

                # Find latest merged parquet file
                latest_parquet = max(
                    DATA_DIR.glob("df_merged_*.parquet"),
                    default=None,
                    key=lambda p: p.stat().st_mtime
                )

                if not latest_parquet:
                    raise FileNotFoundError("No parquet files found in DATA_DIR. Run full DAG pipeline first or check file paths.")

                print(f"📂 Loading from file: {latest_parquet.name}")
                features_df = pd.read_parquet(latest_parquet)
                print(f"✅ Loaded {len(features_df)} rows from parquet")
            else:
                features_df = engineered_data["merged_df"]
                print("✅ Using engineered_data from pipeline")

            print(f"✅ Loaded merged + engineered data: {len(features_df)} rows")
            print(f"   Has predictions: {'PredictedClass' in features_df.columns}")
            print(f"   Has confidence: {'Confidence' in features_df.columns}")

            # ===================================
            # MAP COLUMNS TO SQL TABLE
            # ===================================
            print("\n📋 Mapping columns to SQL schema...")

            sql_insert_df = pd.DataFrame()

            # ✅ Direct mappings - RunDate from predict_date variable
            sql_insert_df['RunDate'] = run_date
            print(f"✅ RunDate set to: {run_date} (from predict_date variable)")

            # MembershipID (uniqueidentifier)
            if 'MembershipID' in features_df.columns:
                sql_insert_df['MembershipID'] = features_df['MembershipID'].astype(str)
            else:
                sql_insert_df['MembershipID'] = None

            # MemberId (nvarchar)
            if 'member_id' in features_df.columns:
                sql_insert_df['MemberId'] = features_df['member_id'].astype(str)
            else:
                sql_insert_df['MemberId'] = None

            # Gender
            sql_insert_df['Gender'] = features_df.get('Gender', None)

            # Age (int)
            if 'Age' in features_df.columns:
                sql_insert_df['Age'] = pd.to_numeric(features_df['Age'], errors='coerce').astype('Int64')
            else:
                sql_insert_df['Age'] = None

            # SubCategory
            sql_insert_df['SubCategory'] = features_df.get('SubCategory', None)

            # Channel
            sql_insert_df['Channel'] = features_df.get('Channel', None)

            # ClubName
            sql_insert_df['ClubName'] = features_df.get('ClubName', None)

            # MembershipTypeDesc
            sql_insert_df['MembershipTypeDesc'] = features_df.get('MembershipTypeDesc', None)

            # Term
            sql_insert_df['Term'] = features_df.get('Term', None)

            # TermDays (int)
            if 'TermDays' in features_df.columns:
                sql_insert_df['TermDays'] = pd.to_numeric(features_df['TermDays'], errors='coerce').astype('Int64')
            else:
                sql_insert_df['TermDays'] = None

            # Lump Sum Flag (bit/int)
            if 'Lump Sum Flag' in features_df.columns:
                sql_insert_df['Lump Sum Flag'] = pd.to_numeric(features_df['Lump Sum Flag'], errors='coerce').astype('Int64')
            else:
                sql_insert_df['Lump Sum Flag'] = None

            # RegularPayment (float)
            if 'RegularPayment' in features_df.columns:
                sql_insert_df['RegularPayment'] = pd.to_numeric(features_df['RegularPayment'], errors='coerce').astype('float64')
            else:
                sql_insert_df['RegularPayment'] = None

            # Base Amount (float)
            if 'Base Amount' in features_df.columns:
                sql_insert_df['Base Amount'] = pd.to_numeric(features_df['Base Amount'], errors='coerce').astype('float64')
            else:
                sql_insert_df['Base Amount'] = None

            # TenureDays (int)
            if 'TenureDays' in features_df.columns:
                sql_insert_df['TenureDays'] = pd.to_numeric(features_df['TenureDays'], errors='coerce').astype('Int64')
            else:
                sql_insert_df['TenureDays'] = None

            # DaysSinceOriginalStart (int)
            if 'DaysSinceOriginalStart' in features_df.columns:
                sql_insert_df['DaysSinceOriginalStart'] = pd.to_numeric(features_df['DaysSinceOriginalStart'], errors='coerce').astype('Int64')
            else:
                sql_insert_df['DaysSinceOriginalStart'] = None

            # DaysSinceLastAccessed (int)
            if 'DaysSinceLastAccessed' in features_df.columns:
                sql_insert_df['DaysSinceLastAccessed'] = pd.to_numeric(features_df['DaysSinceLastAccessed'], errors='coerce').astype('Int64')
            else:
                sql_insert_df['DaysSinceLastAccessed'] = None

            # TotalAttendanceToDate (int)
            if 'TotalAttendanceToDate' in features_df.columns:
                sql_insert_df['TotalAttendanceToDate'] = pd.to_numeric(features_df['TotalAttendanceToDate'], errors='coerce').astype('Int64')
            else:
                sql_insert_df['TotalAttendanceToDate'] = None

            # Visits_Last90d (int)
            if 'Visits_Last90d' in features_df.columns:
                sql_insert_df['Visits_Last90d'] = pd.to_numeric(features_df['Visits_Last90d'], errors='coerce').astype('Int64')
            else:
                sql_insert_df['Visits_Last90d'] = None

            # Visits_Last30d (int)
            if 'Visits_Last30d' in features_df.columns:
                sql_insert_df['Visits_Last30d'] = pd.to_numeric(features_df['Visits_Last30d'], errors='coerce').astype('Int64')
            else:
                sql_insert_df['Visits_Last30d'] = None

            # EndedWithinCoolingPeriod (int)
            if 'EndedWithinCoolingPeriod' in features_df.columns:
                sql_insert_df['EndedWithinCoolingPeriod'] = pd.to_numeric(features_df['EndedWithinCoolingPeriod'], errors='coerce').astype('Int64')
            else:
                sql_insert_df['EndedWithinCoolingPeriod'] = None

            # EWS_Pct (float) - from ews_pct
            if 'ews_pct' in features_df.columns:
                sql_insert_df['EWS_Pct'] = pd.to_numeric(features_df['ews_pct'], errors='coerce').astype('float64')
            else:
                sql_insert_df['EWS_Pct'] = None

            # Risk_Band (nvarchar) - from risk_band
            if 'risk_band' in features_df.columns:
                sql_insert_df['Risk_Band'] = features_df['risk_band'].astype(str)
            else:
                sql_insert_df['Risk_Band'] = None

            # Engineered features (if available, else NULL)
            engineered_features = [
                'Engagement_Rate', 'Recent_Activity_Ratio', 'Declining_Engagement',
                'Monthly_Avg_Visits', 'Payment_to_Attendance_Ratio', 'Recent_Payment_to_Visits',
                'Tenure_Quartile', 'Early_Churn_Risk', 'Inactivity_Ratio',
                'Attendance_Dropoff', 'Access_Gap_Months', 'High_Inactivity'
            ]

            for feature in engineered_features:
                if feature in features_df.columns:
                    sql_insert_df[feature] = pd.to_numeric(features_df[feature], errors='coerce')
                else:
                    sql_insert_df[feature] = None

            # Model predictions (from merged data)
            if 'PredictedClass' in features_df.columns:
                sql_insert_df['Prediction'] = pd.to_numeric(features_df['PredictedClass'], errors='coerce').astype('int64')
            else:
                sql_insert_df['Prediction'] = None

            if 'Confidence' in features_df.columns:
                sql_insert_df['PredictionConfidence'] = pd.to_numeric(features_df['Confidence'], errors='coerce').astype('float64')
            else:
                sql_insert_df['PredictionConfidence'] = None

            # EndDate (always NULL per requirement)
            sql_insert_df['EndDate'] = None

            print(f"✅ Mapped {len(sql_insert_df.columns)} columns")
            print(f"✅ Prepared {len(sql_insert_df)} rows for insert")

            # ===================================
            # CONVERT NA/NaN TO NONE (SQL NULL)
            # ===================================
            print("\n🔧 Converting NA/NaN to SQL NULL...")

            # ✅ SAVE RunDate BEFORE conversion (to prevent it from being set to NULL)
            run_date_values = sql_insert_df['RunDate'].copy()
            print(f"✅ Saved RunDate: {run_date_values.iloc[0] if len(run_date_values) > 0 else 'empty'}")

            # Replace all pd.NA and NaN with None for SQL compatibility
            sql_insert_df = sql_insert_df.where(pd.notna(sql_insert_df), None)

            # Also handle pd.NA explicitly
            for col in sql_insert_df.columns:
                sql_insert_df[col] = sql_insert_df[col].apply(
                    lambda x: None if pd.isna(x) else x
                )

            # ✅ RESTORE RunDate with original values (prevent NULL override)
            sql_insert_df['RunDate'] = run_date_values
            print(f"✅ Restored RunDate: {sql_insert_df['RunDate'].iloc[0] if len(sql_insert_df) > 0 else 'empty'}")
            print("✅ NA/NaN converted to NULL (RunDate preserved)")

            # ===================================
            # INSERT INTO SQL (Using SQLAlchemy for better SQL Server support)
            # ===================================
            print("\n📤 Inserting into SQL table...")
            print(f"📊 Total records to insert: {len(sql_insert_df)}")

            from sqlalchemy import create_engine, text

            # Get connection string from MsSql hook
            conn = hook.get_conn()
            conn_str = hook.get_uri()

            # Create SQLAlchemy engine
            engine = create_engine(conn_str)

            table_name = "repo.MembershipRetentionPredictions"

            # ✅ Batch insert with progress logging
            batch_size = 100
            total_rows = len(sql_insert_df)
            inserted_count = 0

            for i in range(0, total_rows, batch_size):
                batch_df = sql_insert_df.iloc[i : i + batch_size]

                try:
                    batch_df.to_sql(
                        table_name.split('.')[-1],  # table name
                        engine,
                        schema=table_name.split('.')[0] if '.' in table_name else 'dbo',  # schema
                        if_exists='append',
                        index=False,
                        method='multi'
                    )

                    inserted_count += len(batch_df)
                    progress_pct = (inserted_count / total_rows) * 100

                    # ✅ Log every 100 rows
                    print(f"✅ Inserted {inserted_count}/{total_rows} rows ({progress_pct:.1f}%)")

                except Exception as e:
                    print(f"❌ Batch {i}-{i+batch_size} failed: {e}")
                    raise

            conn.close()
            engine.dispose()

            print(f"\n✅ Successfully inserted {inserted_count}/{total_rows} records (100.0%)")

            # ===================================
            # SUMMARY STATS
            # ===================================
            print("\n📊 Insert Summary:")
            print(f"   Total records: {len(sql_insert_df)}")
            print(f"   Inserted: {inserted_count}")
            print(f"   Failed: {len(sql_insert_df) - inserted_count}")
            print(f"   Table: {table_name}")
            print(f"   Columns: {len(sql_insert_df.columns)}")

            return {
                "status": "success",
                "records_inserted": inserted_count,
                "total_records": len(sql_insert_df),
                "table": table_name,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            print(f"❌ SQL insert failed: {str(e)}")
            raise Exception(f"Insert to SQL failed: {e}")

    # Flow
    # read_data → snapshots → ingest → model_predict → JOIN → engineer_features → insert_sql
    raw_df = read_data_task()
    snap_df = snapshots_task(raw_df)
    data_paths = ingest_task(snap_df)
    model_output = model_task(data_paths)
    merge_output = merge_predictions_task(model_output)
    engineered_output = engineer_features_task(merge_output)
    insert_sql_task(engineered_output)
