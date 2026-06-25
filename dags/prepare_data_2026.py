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
TRAIN_DATE = "2026-01-01"


# ================================
# SAVE ALL CLASSES 1, 2 & 3
# ================================
def save_all_classes(data_paths: dict):
    """
    Load df_merged and keep all classes (1, 2, and 3).
    Save as df_merged_2026.parquet
    """
    try:
        df = pd.read_parquet(data_paths["df_merged_path"])

        # Save all data as-is
        output_path = DATA_DIR / "df_merged_2026.parquet"
        df.to_parquet(output_path, index=False)

        print(f"✅ Prepared 2026 data: {len(df)} rows (all classes 1, 2 & 3)")
        print(f"📦 Saved to {output_path}")

        return {"df_merged_2026_path": str(output_path)}

    except Exception as e:
        raise Exception(f"Preparing 2026 data failed: {e}")


# ================================
# DAG
# ================================
with DAG(
    dag_id="PrepareData2026",
    description="Prepare training data for 2026-01-01 (all classes 1, 2 & 3)",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False
) as dag:

    # Read data for 2026-01-01
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
    def save_task(data_paths):
        return save_all_classes(data_paths)

    # Flow
    raw_df = read_data_task()
    snap_df = snapshots_task(raw_df)
    data_paths = ingest_task(snap_df)
    save_task(data_paths)
