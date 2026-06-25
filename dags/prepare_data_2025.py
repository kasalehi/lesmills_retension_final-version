from src.les.train.run import snapshot as snapshot_fn
from src.les.train.readdata import read_data as read_data_fn
from src.les.train.ingestfortrain import ingest_data
from airflow import DAG
from airflow.sdk import task
from airflow.models import Variable
from datetime import datetime
from pathlib import Path

import pandas as pd

# ================================
# PATHS
# ================================
DATA_DIR = Path("/usr/local/airflow/include/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ================================
# HARDCODED DATE
# ================================
TRAIN_DATE = "2025-01-01"


# ================================
# FILTER FOR CLASSES 1 & 2
# ================================
def filter_classes_1_2(data_paths: dict):
    """
    Load df_merged and filter to only include Churned classes 1 and 2.
    Save as df_merged_2025.parquet
    """
    try:
        df = pd.read_parquet(data_paths["df_merged_path"])

        # Filter for classes 1 and 2 only
        df_filtered = df[df["Churned"].isin([1, 2])].copy()

        # Save filtered data
        output_path = DATA_DIR / "df_merged_2025.parquet"
        df_filtered.to_parquet(output_path, index=False)

        print(f"✅ Filtered 2025 data: {len(df_filtered)} rows (classes 1 & 2)")
        print(f"📦 Saved to {output_path}")

        return {"df_merged_2025_path": str(output_path)}

    except Exception as e:
        raise Exception(f"Filtering 2025 data failed: {e}")


# ================================
# DAG
# ================================
with DAG(
    dag_id="PrepareData2025",
    description="Prepare balanced training data for 2025-01-01 (classes 1 & 2)",
    schedule=None,
    start_date=datetime(2025, 1, 1),
    catchup=False
) as dag:

    # Read data for 2025-01-01
    @task
    def read_data_task():
        Variable.set("train_date", TRAIN_DATE)
        return read_data_fn()

    @task
    def snapshots_task(df):
        return snapshot_fn(df)

    @task
    def ingest_task(snap_df):
        return ingest_data(snap_df)

    @task
    def filter_task(data_paths):
        return filter_classes_1_2(data_paths)

    # Flow
    raw_df = read_data_task()
    snap_df = snapshots_task(raw_df)
    data_paths = ingest_task(snap_df)
    filter_task(data_paths)
