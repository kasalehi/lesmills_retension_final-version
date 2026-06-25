import streamlit as st

st.title("📘 Les Mills Membership Prediction App")

# --- Main Documentation ---
with st.expander("📖 About This App"):
    st.markdown("""
## Overview
This application is an end-to-end MLOps solution developed by **Keyvan** for the **Les Mills National Office**.

It enables tracking and prediction of membership behaviour using machine learning models.

---

## 🎯 Purpose
- Predict membership risk
- Identify churn likelihood
- Support retention strategies

---

## 🤖 Models Used

### CatBoost
- Generates:
  - `ews_pct` (Early Warning Score)
  - `risk_band`

### XGBoost
- Main prediction model

### Random Forest
- Benchmark model

---

## 🔄 Data Flow
1. Upload dataset
2. Apply CatBoost → generate `ews_pct` and `risk_band`
3. Feed enriched data into prediction models
4. Display results in dashboard

---

## 📥 Required Input Columns
- MembershipID  
- SubCategory  
- RegularPayment  
- Gender  
- Age  
- TotalAttendance  

---

## 📤 Output Columns
- ews_pct  
- risk_band  
- prediction  
- prediction_confidence  

---

## ⚙️ Deployment
- Docker containerised application  
- Portainer → Port 9000  
- Streamlit → Port 8501  

---

## 🛠 Support
For any issues or improvements, please contact the IT team.

---

Enjoy your predictions 🚀
""")

# --- Optional extra sections (clean UX) ---
with st.expander("📥 Input Example"):
    st.code("""
MembershipID,SubCategory,RegularPayment,Gender,Age,TotalAttendance
1001,Premium,45.0,Male,32,120
""", language="csv")

with st.expander("⚠️ Important Notes"):
    st.markdown("""
- Ensure column names match exactly  
- No missing values in required fields  
- Model performance depends on data quality  
""")