from src.les.train.train import ModelTraing
from airflow import DAG
from airflow.sdk import task
from datetime import datetime
from pathlib import Path

import pandas as pd
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
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
# TRANSFORM + TRAIN WITH BALANCED DATA
# ================================
def transform_and_train_balanced():
    try:
        # ✅ Load balanced merged parquet
        df = pd.read_parquet(BALANCED_DATA_PATH)

        print(f"📊 Loaded balanced data: {len(df)} rows")
        print(f"📊 Class distribution:\n{df['Churned'].value_counts().sort_index()}")

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
        print(f"   Features in X: {list(X.columns)[:10]}... (showing first 10)")
        print(f"   Target values: {set(y)}")

        # Check if target is still in X
        if TARGET in X.columns:
            print(f"   ❌ WARNING: Target '{TARGET}' found in features!")
        if 'Churned' in X.columns:
            print(f"   ❌ WARNING: 'Churned' column found in features!")

        # Check for duplicate columns
        if X.shape[1] != len(X.columns):
            print(f"   ❌ WARNING: Duplicate column names found!")

        # ===================================
        # DROP COLUMNS WITH DATA LEAKAGE + WEAK FEATURES
        # ===================================
        # Remove data leakage columns
        leakage_cols = ['DaysToContractEnd','end_date']  # 0.866 correlation = direct proxy for churn date

        pause_cols = ['WeeksPausedToDate','PauseCount','TotalDaysPaused','CurrentlyPaused','DaysSinceLastPauseEnded','LongestPauseDuration','AvgPauseDuration']
        covid_cols = ['COVID19_PauseCount']
        weak_cols = ['Discount','DaysSinceLastVisit']
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
        # CREATE PREPROCESSOR - AUTO-DETECT TYPES + IMPUTATION
        # ===================================
        # Auto-detect numeric and categorical columns
        numeric_features = X.select_dtypes(include=['int64', 'int32', 'float64', 'float32']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'string', 'category']).columns.tolist()

        print(f"📊 Numeric features ({len(numeric_features)}): {numeric_features}")
        print(f"📊 Categorical features ({len(categorical_features)}): {categorical_features}")
        print(f"🔍 Missing values in numeric: {X[numeric_features].isnull().sum().sum()}")
        print(f"🔍 Missing values in categorical: {X[categorical_features].isnull().sum().sum()}")

        # Create preprocessor with imputation + scaling/encoding
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),  # Fill NaN with median
                    ('scaler', StandardScaler())
                ]), numeric_features),
                ('cat', Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),  # Fill NaN with most common
                    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
                ]), categorical_features)
            ],
            remainder='drop'
        )

        # Split
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train model
        model_obj = ModelTraing()
        trained_model, best_model_name, results = model_obj.trainingModel(
            x_train, y_train, x_test, y_test, preprocessor
        )

        # ===================================
        # INVESTIGATE PERFECT ACCURACY
        # ===================================
        print(f"\n🔍 MODEL INSPECTION:")
        print(f"   Best model: {best_model_name}")

        # Get feature importance if available
        if hasattr(trained_model, 'named_steps'):
            clf = trained_model.named_steps['clf']
            if hasattr(clf, 'feature_importances_'):
                top_features = sorted(enumerate(clf.feature_importances_), key=lambda x: x[1], reverse=True)[:5]
                print(f"   Top 5 important features:")
                for idx, importance in top_features:
                    print(f"      Feature {idx}: importance = {importance:.4f}")

        # Check train vs test split
        print(f"   Train size: {len(x_train)}, Test size: {len(x_test)}")
        print(f"   Train/Test overlap: {len(set(range(len(x_train))) & set(range(len(x_train), len(x_train) + len(x_test))))}")

        # =========================
        # EVALUATION
        # =========================
        y_pred = trained_model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"✅ Model Accuracy (Balanced): {accuracy:.4f}")

        # Calculate Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)

        # Calculate Classification Report
        class_names = ['Class 0 (Churn 0-3mo)', 'Class 1 (Churn 3-6mo)', 'Class 2 (No Churn)']
        class_report = classification_report(y_test, y_pred, target_names=class_names, digits=4)

        # Per-class metrics
        report_dict = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)

        print(f"\n📊 CONFUSION MATRIX:")
        print(cm)
        print(f"\n📋 CLASSIFICATION REPORT:")
        print(class_report)

        # =========================
        # SAVE ARTIFACTS
        # =========================
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save model
        model_path = ARTIFACT_DIR / f"model_balanced_{timestamp}.pkl"
        joblib.dump(trained_model, model_path)

        # Save metrics
        metrics = {
            "accuracy": float(accuracy),
            "rows": len(df),
            "timestamp": timestamp,
            "model_type": "balanced",
            "class_distribution": df['Churned'].value_counts().sort_index().to_dict()
        }

        metrics_path = ARTIFACT_DIR / f"metrics_balanced_{timestamp}.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)

        # Save log with detailed metrics
        log_path = ARTIFACT_DIR / f"log_balanced_{timestamp}.txt"
        with open(log_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("MODEL TRAINING REPORT - BALANCED DATA\n")
            f.write("=" * 80 + "\n\n")

            f.write("BASIC INFORMATION:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Model Type: Balanced (2024 + 2025 + 2026)\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Total Rows: {len(df)}\n")
            f.write(f"Test Set Size: {len(x_test)}\n")
            f.write(f"Train Set Size: {len(x_train)}\n\n")

            f.write("OVERALL METRICS:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n\n")

            f.write("CONFUSION MATRIX:\n")
            f.write("-" * 80 + "\n")
            f.write("Rows: True Labels | Columns: Predicted Labels\n")
            f.write("       Class 0  Class 1  Class 2\n")
            for i, row in enumerate(cm):
                f.write(f"Class {i}:  {row[0]:6d}  {row[1]:6d}  {row[2]:6d}\n")
            f.write("\n")

            f.write("CLASSIFICATION REPORT:\n")
            f.write("-" * 80 + "\n")
            f.write(class_report)
            f.write("\n")

            f.write("PER-CLASS METRICS:\n")
            f.write("-" * 80 + "\n")
            for class_name in class_names:
                if class_name in report_dict:
                    metrics = report_dict[class_name]
                    f.write(f"\n{class_name}:\n")
                    f.write(f"  Precision: {metrics.get('precision', 0):.4f}\n")
                    f.write(f"  Recall:    {metrics.get('recall', 0):.4f}\n")
                    f.write(f"  F1-Score:  {metrics.get('f1-score', 0):.4f}\n")
                    f.write(f"  Support:   {int(metrics.get('support', 0))}\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("CLASS DISTRIBUTION:\n")
            f.write("-" * 80 + "\n")
            class_dist = pd.Series(y_test).value_counts().sort_index()
            for class_label, count in class_dist.items():
                pct = (count / len(y_test)) * 100
                f.write(f"Class {class_label}: {count:6d} ({pct:6.2f}%)\n")
            f.write("\n")

        print("📦 Artifacts saved (Balanced):")
        print(model_path)
        print(metrics_path)
        print(log_path)

        return {
            "model_path": str(model_path),
            "metrics_path": str(metrics_path),
            "log_path": str(log_path),
            "accuracy": accuracy,
            "model_type": "balanced"
        }

    except Exception as e:
        raise Exception(f"Training balanced model failed: {e}")


# ================================
# DAG
# ================================
with DAG(
    dag_id="ModelTrainBalanced",
    description="Les Mills train model pipeline with balanced data (2024+2025+2026)",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False
) as dag:

    @task
    def train_task():
        return transform_and_train_balanced()

    train_task()
