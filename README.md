# 🏥 DiabetesAI — Diabetes Risk Prediction App

An end-to-end, data-driven machine learning application that predicts diabetes diagnosis risk from patient demographic, lifestyle, and clinical data. Built with **scikit-learn**, **XGBoost**, and **Streamlit**, and deployed live on Streamlit Community Cloud.

🔗 **Live App:** [add your Streamlit Cloud link here]
📊 **Dataset:** 100,000 patient records, 31 features

---

## Overview

This app trains and compares three classification models to predict whether a patient is likely to be diagnosed with diabetes, based on demographic information (age, gender, ethnicity), lifestyle factors (physical activity, diet, sleep, smoking), and clinical measurements (BMI, blood pressure, cholesterol panel, glucose levels, HbA1c).

Rather than hardcoding column names, the pipeline is **schema-aware**: it automatically detects categorical vs. numerical features, lets the user pick the target column, and flags likely **data leakage columns** before training.

### Why this matters

The raw dataset includes two columns — `diabetes_risk_score` and `diabetes_stage` — that are derived *from* the diagnosis itself rather than being independent predictors. Training on them inflates accuracy to a meaningless ~99%+. This app detects and excludes them by default, which is why the reported accuracy below is realistic rather than misleading.

---

## Model Performance

Trained on an 80/20 split of the 100,000-row dataset, with leakage columns excluded:

| Model | Accuracy |
|---|---|
| Random Forest | 92.0% |
| XGBoost | 91.5% |
| Logistic Regression | 86.1% |
| Model               | Accuracy |
| ------------------- | -------- |
| Random Forest       | 92.0%    |
| XGBoost             | 91.5%    |
| Logistic Regression | 86.1%    |

---

## Features

- 📊 **Data Analysis** — dataset summary statistics, target distribution, correlation heatmap
- 🤖 **Model Training** — trains Logistic Regression, Random Forest, and XGBoost classifiers via a single `scikit-learn` `Pipeline` (handles categorical encoding + scaling automatically, avoiding train/inference mismatch bugs)
- ⚠️ **Leakage detection** — flags columns likely derived from the target and excludes them by default
- 🔮 **Live prediction** — dynamically generated input form based on the dataset's actual columns, with real-time prediction and probability score
- 🔧 **Self-diagnosing debug panel** — surfaces the deployed environment's actual file structure if the dataset can't be loaded, for fast troubleshooting on cloud deployments

---

## Tech Stack

- **Frontend/App:** Streamlit
- **ML:** scikit-learn (Logistic Regression, Random Forest, ColumnTransformer, Pipeline), XGBoost
- **Data:** pandas, NumPy
- **Visualization:** Matplotlib, Seaborn

---

## Project Structure

```
DiabetesAI.app/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── data/
│   └── diabetes_dataset.csv
└── README.md
```

---

## Running Locally

```bash
# Clone the repo
git clone https://github.com/Waseem771/DiabetesAI.app.git
cd DiabetesAI.app

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## Deployment

This app is deployed on [Streamlit Community Cloud](https://share.streamlit.io):

1. Push this repo to GitHub (already done ✅)
2. On Streamlit Cloud, connect the repo and set **Main file path** to `app.py`
3. Deploy — the dataset is loaded via a path relative to `app.py`, so no manual configuration is needed

---

## Dataset

The dataset (`data/diabetes_dataset.csv`) contains 100,000 anonymized patient records with 31 features spanning:

- **Demographics:** age, gender, ethnicity, education level, income level, employment status
- **Lifestyle:** smoking status, alcohol consumption, physical activity, diet score, sleep hours, screen time
- **Medical history:** family history of diabetes, hypertension history, cardiovascular history
- **Clinical measurements:** BMI, waist-to-hip ratio, blood pressure, heart rate, cholesterol panel (total/HDL/LDL), triglycerides, fasting/postprandial glucose, insulin level, HbA1c
- **Target:** `diagnosed_diabetes` (0 = not diagnosed, 1 = diagnosed)

> `diabetes_risk_score` and `diabetes_stage` are excluded from training by default as they are derived from the diagnosis outcome (see *Why this matters* above).
> `diabetes_risk_score` and `diabetes_stage` are excluded from training by default as they are derived from the diagnosis outcome (see _Why this matters_ above).

---

## Disclaimer

This app is built for educational and portfolio purposes. It is **not a medical diagnostic tool** and should not be used for actual clinical decision-making.

---

## Author

**Waseem Hassan**
AI/ML Engineer | Network Infrastructure → Applied ML
[LinkedIn](https://linkedin.com/in/waseemhassanshk) · [GitHub](https://github.com/Waseem771) · [Kaggle](https://kaggle.com/waseem7711)[LinkedIn](https://linkedin.com/in/waseemhassanshk) · [GitHub](https://github.com/Waseem771) · [Kaggle](https://kaggle.com/waseem7711)
