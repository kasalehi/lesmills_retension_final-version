
# Les Mills Retention Prediction Platform

## Overview
This project implements an end-to-end Machine Learning pipeline for predicting customer churn risk using **Apache Airflow (Astro), Docker, and Streamlit**.

The system performs the following workflow:

SQL Server → Airflow Pipeline → Feature Engineering → ML Prediction → Save Results → Streamlit Dashboard

The pipeline runs automatically through Airflow, and the Streamlit application allows users to upload or view datasets and run predictions using the trained model.

---

# Architecture

SQL Server
    ↓
Airflow DAG (Astro + Docker)
    ↓
Feature Engineering & Data Merge
    ↓
Load Trained Model
    ↓
Save Predictions Dataset
    ↓
Streamlit Dashboard

---

# Project Structure

```
lesmills-ml-pipeline
│
├── dags/
│   └── lesmills_retention.py
│
├── src/
│   └── les/
│       ├── ingest.py
│       ├── transform.py
│       ├── train.py
│       ├── utils.py
│       ├── predict.py
│       └── logger.py
│
├── include/
│   ├── artifacts/
│   │   ├── model.pkl
│   │   └── preprocessor.pkl
│   │
│   ├── data/
│   │   └── predictions.csv
│   │
│   └── lesmills_project/
│       └── streamlit.py
│
├── requirements.txt
├── Dockerfile
├── airflow_settings.yaml
└── README.md
```

---

# Prerequisites

Install the following:

- Docker
- Astronomer CLI
- Git
- Python 3.10+

Install Astronomer CLI:
https://docs.astronomer.io/astro/cli/install-cli

Verify installation:

```
astro version
```

---

# Running the Project Locally

## Step 1 — Clone the Repository

```
git clone https://github.com/kasalehi/lesmills.git
cd lesmills
```

---

# Step 2 — Start Airflow with Astro

Run:

```
astro dev start
```

This starts Docker containers for:

- Airflow Scheduler
- Airflow Webserver
- DAG Processor
- Triggerer
- PostgreSQL metadata database

Airflow UI will be available at:

http://localhost:8080

Default credentials:

username: admin  
password: admin

---

# Step 3 — Trigger the Pipeline

Open the Airflow UI.

Navigate to the DAG:

```
lesmills
```

Click **Trigger DAG**.

The pipeline will perform:

1. Read data from SQL Server
2. Generate snapshots
3. Merge dataset
4. Load trained ML model
5. Run predictions
6. Save results

Predictions are saved in:

```
include/data/predictions.csv
```

---

# Step 4 — Run the Streamlit Application

Open a terminal and run:

```
astro dev bash
```

Then start Streamlit:

```
streamlit run include/lesmills_project/streamlit.py --server.port 8501 --server.address 0.0.0.0
```

Open the Streamlit dashboard:

http://localhost:8501

---

# Using the Streamlit Dashboard

The Streamlit application allows users to:

- Upload a CSV dataset
- Run predictions using the trained model
- View prediction results
- Download prediction outputs

The model and preprocessing pipeline are loaded from:

```
include/artifacts/model.pkl
include/artifacts/preprocessor.pkl
```

---

# Data Flow

Airflow DAG
    ↓
Extract data from SQL Server
    ↓
Create snapshot dataset
    ↓
Merge features
    ↓
Load trained ML model
    ↓
Predict churn probability
    ↓
Save predictions.csv
    ↓
Streamlit reads predictions

---

# Model Training (Optional)

The project initially trains a model using:

- Random Forest
- Gradient Boosting
- GridSearchCV

Artifacts are stored in:

```
include/artifacts/
```

Files:

```
model.pkl
preprocessor.pkl
```

Once the model is trained, the pipeline switches to **prediction mode**.

---

# Environment Variables

Secrets such as service credentials should **not be committed to Git**.

Use environment variables or `.env` files instead.

Example:

```
GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
```

---

# Development Workflow

Typical workflow for developers:

1. Start Astro
2. Trigger Airflow DAG
3. Generate predictions dataset
4. Launch Streamlit UI
5. Upload dataset or view predictions

---

# Useful Commands

Start Airflow:

```
astro dev start
```

Stop Airflow:

```
astro dev stop
```

Restart Airflow:

```
astro dev restart
```

Enter container:

```
astro dev bash
```

Run Streamlit:

```
streamlit run include/lesmills_project/streamlit.py
```

---

# Future Improvements

Possible enhancements:

- Model versioning
- Prediction monitoring
- Automated model retraining
- Feature drift detection
- Streamlit analytics dashboards

---

# Author

Keyvan Salehi 2025/11/12
Les Mills New Zealand