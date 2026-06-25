import streamlit as st
import pandas as pd
import dill
from pathlib import Path

st.set_page_config(page_title="Main", initial_sidebar_state="expanded")
st.title("🏋️ Les Mills Membership Prediction App")
# -------- Load pipeline --------
@st.cache_resource
def load_pipeline():
    artifacts_dir = Path("/usr/local/airflow/include/artifacts")
    model_path = artifacts_dir / "model.pkl"

    if not model_path.exists():
        st.error(f"Model not found at {model_path}")
        st.stop()

    with open(model_path, "rb") as f:
        pipeline = dill.load(f)

    return pipeline


# -------- Streamlit UI --------
def run_app():

    pipeline = load_pipeline()

    ID_COL = "MembershipID"

    FEATURE_COLS = [
        "SubCategory",
        "RegularPayment",
        "Gender",
        "Age",
        "TotalAttendance",
        "ews_pct",
        "risk_band",
    ]

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:

        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            st.stop()

        st.subheader("Preview of uploaded data")
        st.dataframe(df.head())

        required_cols = [ID_COL] + FEATURE_COLS
        missing_cols = [c for c in required_cols if c not in df.columns]

        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            st.stop()

        if st.button("Run prediction"):

            try:
                X = df[FEATURE_COLS]

                y_pred = pipeline.predict(X)

                df_result = df[[ID_COL]].copy()
                df_result["prediction"] = y_pred

                if hasattr(pipeline, "predict_proba"):
                    proba = pipeline.predict_proba(X)
                    df_result["confidence"] = proba.max(axis=1)

                st.subheader("Prediction results")
                st.dataframe(df_result.head())

                csv_out = df_result.to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="Download results CSV",
                    data=csv_out,
                    file_name="predictions.csv",
                    mime="text/csv",
                )

            except Exception as e:
                st.error(f"Prediction error: {e}")

    else:
        st.info("Upload a CSV file to start predictions.")


if __name__ == "__main__":
    run_app()