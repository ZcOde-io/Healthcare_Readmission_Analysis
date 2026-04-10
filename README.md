# 🏥 Healthcare Analytics — Diabetes Hospital Readmission Predictor

Predicts **30-day hospital readmission risk** for diabetes patients using machine learning.

## Project Structure
```
readmission_app/
├── main.py                              ← FastAPI web application
├── Healthcare_Readmission_Analysis.ipynb ← Full EDA & ML notebook
├── requirements.txt
├── data/
│   └── diabetes_readmission.csv         ← 8,000-row dataset
├── models/
│   ├── best_model.pkl                   ← Trained Random Forest
│   ├── scaler.pkl                       ← StandardScaler
│   ├── feature_cols.pkl                 ← Feature names
│   └── metrics.json                     ← Model comparison results
├── static/
│   └── analysis.png                     ← EDA + evaluation charts
└── templates/
    ├── index.html                        ← Main prediction UI
    └── history.html                      ← Prediction history
```

## Setup & Run

```bash
# Install dependencies
pip install -r requirements.txt

# Start the web app
uvicorn main:app --reload --port 8000

# Open in browser
# http://localhost:8000
```

## Run Jupyter Notebook

```bash
jupyter notebook Healthcare_Readmission_Analysis.ipynb
```

## ML Pipeline

| Step | Details |
|------|---------|
| Dataset | 8,000 records, 31.5% readmission rate |
| Imbalance | SMOTE oversampling |
| Models | Logistic Regression, Random Forest ✅ (best), XGBoost |
| Best AUC | 0.607 (Random Forest) |
| Features | 20 total (17 clinical + 3 engineered) |

## Key Features Used
- **Prior visits** — inpatient, emergency, outpatient history
- **HbA1c result** — blood sugar control indicator
- **Time in hospital** — length of current stay
- **Medications** — number & changes made
- **Admission type** — emergency vs elective
- **Demographics** — age, gender, race

## Web App Features
- Patient risk assessment form (17 inputs)
- Real-time prediction with probability score
- SQLite database storing all predictions
- Model comparison dashboard
- EDA analysis charts
- Full prediction history page
