from pathlib import Path
import pandas as pd
from include.lesmills_project.app import build_snapshots, score_catboost, build_outreach
from airflow.models import Variable
from include.lesmills_project.ingest import read_data
def snapshot(df: pd.DataFrame) -> pd.DataFrame:
    active_q, paused_q, scores, model, feature_cols = build_outreach(
        df,
        k_weeks=6,
        capacity=500,
    )
    scores = score_catboost(model, feature_cols, k_weeks=6, df=df)
    snapshots = build_snapshots(scores, df)

    snapshots["main_reason"] = snapshots["main_reason"].replace({
        "erratic_usage": "Irregular_Movement",
        "drought_streak": "Continously_Inactive",
    })
    snapshots["reasons"] = snapshots["reasons"].replace({
        "erratic_usage": "Irregular_Movement",
        "drought_streak": "Continously_Inactive",
    })

    return snapshots
