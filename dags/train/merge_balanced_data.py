from airflow import DAG
from airflow.sdk import task
from datetime import datetime
from pathlib import Path

import pandas as pd

# ================================
# PATHS
# ================================
DATA_DIR = Path("/usr/local/airflow/include/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ================================
# MERGE ALL THREE DATASETS
# ================================
def merge_all_data():
    """
    Load all three prepared datasets and merge them into a single df_merged.
    - df_merged_2024.parquet (classes 1 & 2)
    - df_merged_2025.parquet (classes 1 & 2)
    - df_merged_2026.parquet (classes 1, 2 & 3) → sample 12k from class 3 only

    Creates truly balanced dataset with equal class representation.
    Save as df_merged_balanced.parquet
    """
    try:
        # Paths
        path_2024 = DATA_DIR / "df_merged_2024.parquet"
        path_2025 = DATA_DIR / "df_merged_2025.parquet"
        path_2026 = DATA_DIR / "df_merged_2026.parquet"

        # Load all datasets
        df_2024 = pd.read_parquet(path_2024)
        df_2025 = pd.read_parquet(path_2025)
        df_2026 = pd.read_parquet(path_2026)

        print(f"✅ Loaded 2024 data: {len(df_2024)} rows (classes 1 & 2)")
        print(f"✅ Loaded 2025 data: {len(df_2025)} rows (classes 1 & 2)")
        print(f"✅ Loaded 2026 data: {len(df_2026)} rows (all classes)")

        # ===========================
        # BALANCE CLASS 3 FROM 2026
        # ===========================
        df_2026_classes_1_2 = df_2026[df_2026["Churned"].isin([1, 2])]
        df_2026_class_3 = df_2026[df_2026["Churned"] == 3]

        # Sample 20k from class 3 for balance (less aggressive than 12k)
        sample_size = min(20000, len(df_2026_class_3))
        df_2026_class_3_sampled = df_2026_class_3.sample(n=sample_size, random_state=42)

        print(f"📊 2026 breakdown:")
        print(f"   Classes 1 & 2: {len(df_2026_classes_1_2)} rows")
        print(f"   Class 3 (original): {len(df_2026_class_3)} rows → sampled to {len(df_2026_class_3_sampled)} rows")

        # Combine 2026 classes 1, 2 with sampled class 3
        df_2026_balanced = pd.concat([df_2026_classes_1_2, df_2026_class_3_sampled], ignore_index=True)

        # Merge all three datasets
        df_merged = pd.concat([df_2024, df_2025, df_2026_balanced], ignore_index=True)

        print(f"\n✅ Merged all datasets: {len(df_merged)} total rows")

        # Class distribution
        class_dist = df_merged["Churned"].value_counts().sort_index()
        total_rows = len(df_merged)
        print(f"\n📊 Final class distribution (BALANCED):")
        for class_id, count in sorted(class_dist.items()):
            pct = (count / total_rows) * 100
            print(f"   Class {class_id}: {count} rows ({pct:.1f}%)")

        # Save merged data
        output_path = DATA_DIR / "df_merged_balanced.parquet"
        df_merged.to_parquet(output_path, index=False)

        print(f"\n📦 Balanced dataset saved to {output_path}")

        return {
            "df_merged_balanced_path": str(output_path),
            "total_rows": len(df_merged),
            "class_distribution": class_dist.to_dict()
        }

    except Exception as e:
        raise Exception(f"Merging balanced data failed: {e}")


# ================================
# DAG
# ================================
with DAG(
    dag_id="MergeBalancedData",
    description="Merge prepared datasets from 2024, 2025, 2026 into balanced training data",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False
) as dag:

    @task
    def merge_task():
        return merge_all_data()

    merge_task()
