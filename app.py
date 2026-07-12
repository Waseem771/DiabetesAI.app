import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, r2_score, confusion_matrix, classification_report
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# ============ PAGE CONFIGURATION ============
st.set_page_config(page_title="Diabetes Prediction", page_icon="🏥", layout="wide")

st.title("🏥 Diabetes Prediction Model")
st.markdown("---")

# ============ DATA PATH (RELATIVE — WORKS ON STREAMLIT CLOUD) ============
# On GitHub/Streamlit Cloud there is no G: drive, so we use a path relative
# to this script. Put your CSV at: <repo_root>/data/diabetes_dataset.csv
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(APP_DIR, "data", "diabetes_dataset.csv")

REQUIRED_COLUMNS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]

# Common alternate names seen in different Pima-diabetes CSV variants
# (e.g. the Microsoft Learn / Azure ML version uses PlasmaGlucose, Diabetic, etc.)
COLUMN_ALIASES = {
    "pregnancies": "Pregnancies",
    "glucose": "Glucose",
    "plasmaglucose": "Glucose",
    "glucose_level": "Glucose",
    "bloodpressure": "BloodPressure",
    "blood_pressure": "BloodPressure",
    "diastolicbloodpressure": "BloodPressure",
    "skinthickness": "SkinThickness",
    "skin_thickness": "SkinThickness",
    "tricepsthickness": "SkinThickness",
    "insulin": "Insulin",
    "seruminsulin": "Insulin",
    "bmi": "BMI",
    "diabetespedigreefunction": "DiabetesPedigreeFunction",
    "diabetespedigree": "DiabetesPedigreeFunction",
    "dpf": "DiabetesPedigreeFunction",
    "age": "Age",
    "outcome": "Outcome",
    "diabetic": "Outcome",
    "class": "Outcome",
    "target": "Outcome",
}


@st.cache_data
def load_data(path):
    """
    Load the dataset. Cached so it only runs once.
    Handles two common real-world issues:
    1. Wrong delimiter detection (e.g. semicolon-separated files read as one column)
    2. Column names that don't exactly match REQUIRED_COLUMNS (renamed via aliases)
    """
    if not os.path.exists(path):
        return None, "not_found"

    try:
        df = pd.read_csv(path)
    except Exception:
        return None, "parse_error"

    # If the whole header collapsed into a single column, the delimiter
    # is probably not a comma — retry letting pandas sniff it.
    if df.shape[1] == 1:
        try:
            df = pd.read_csv(path, sep=None, engine="python")
        except Exception:
            pass

    # Normalize column names: strip whitespace, then map known aliases
    # (case-insensitive) onto the exact names the app expects.
    original_columns = list(df.columns)
    cleaned = {c: c.strip() for c in df.columns}
    df = df.rename(columns=cleaned)
    rename_map = {}
    for c in df.columns:
        key = c.strip().lower().replace(" ", "").replace("-", "")
        if key in COLUMN_ALIASES:
            rename_map[c] = COLUMN_ALIASES[key]
    df = df.rename(columns=rename_map)

    df.attrs["original_columns"] = original_columns
    return df, "ok"


def show_debug_info(path):
    """
    Self-diagnosing helper: if the dataset can't be found, show exactly
    what the deployed environment actually looks like instead of making
    the user guess. This is the fastest way to spot LFS-pointer files,
    wrong branches, or stale deploys.
    """
    with st.expander("🔧 Debug info (click to expand)"):
        st.write("**Expected path:**", path)
        st.write("**App directory (`__file__` location):**", APP_DIR)
        st.write("**Files in app directory:**")
        try:
            st.code("\n".join(sorted(os.listdir(APP_DIR))))
        except Exception as e:
            st.write(f"Could not list app directory: {e}")

        data_dir = os.path.join(APP_DIR, "data")
        st.write("**Files in `data/` folder:**")
        if os.path.isdir(data_dir):
            entries = sorted(os.listdir(data_dir))
            st.code("\n".join(entries) if entries else "(empty folder)")
            # Flag a classic Git LFS pointer file: real dataset is ~13MB,
            # an LFS pointer is only ~130 bytes of plain text.
            csv_path = os.path.join(data_dir, "diabetes_dataset.csv")
            if os.path.exists(csv_path):
                size_kb = os.path.getsize(csv_path) / 1024
                st.write(f"**diabetes_dataset.csv size:** {size_kb:.1f} KB")
                if size_kb < 5:
                    st.warning(
                        "⚠️ This file is suspiciously small for a 13+ MB dataset. "
                        "It's likely a **Git LFS pointer file**, not the real CSV. "
                        "Check if `.gitattributes` in your repo has an LFS filter rule, "
                        "and if so, either disable LFS for this file or make sure "
                        "Streamlit Cloud can resolve LFS content."
                    )
        else:
            st.code("(no 'data' folder found here)")


def validate_columns(df):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    return missing


# Load once, reused across all pages
df, load_status = load_data(DATASET_PATH)

# ============ SIDEBAR NAVIGATION ============
page = st.sidebar.radio("Select Page:", ["Home", "Data Analysis", "Model Training", "Make Prediction"])

# Friendly, actionable error if the CSV isn't where it should be —
# this is the #1 reason this kind of app breaks on Streamlit Cloud.
if df is None:
    if load_status == "parse_error":
        st.error(
            "⚠️ Found `data/diabetes_dataset.csv` but couldn't parse it as a CSV. "
            "It may be corrupted or an unresolved Git LFS pointer file."
        )
    else:
        st.error(
            f"⚠️ Dataset not found at `data/diabetes_dataset.csv`.\n\n"
            f"Make sure your GitHub repo looks like this:\n\n"
            f"```\n"
            f"your-repo/\n"
            f"├── app.py\n"
            f"├── requirements.txt\n"
            f"└── data/\n"
            f"    └── diabetes_dataset.csv\n"
            f"```"
        )
    show_debug_info(DATASET_PATH)
    st.stop()

missing_cols = validate_columns(df)
if missing_cols:
    st.error(f"⚠️ Dataset is missing expected columns: {missing_cols}")
    st.write("**Actual columns found in your CSV:**")
    st.code(", ".join(df.attrs.get("original_columns", list(df.columns))))
    st.info(
        "If your column names use different naming (e.g. `PlasmaGlucose` instead of "
        "`Glucose`, or `Diabetic` instead of `Outcome`), either rename them in the CSV "
        "to match the list above, or tell me the actual column names and I'll update "
        "the app's alias mapping to handle them automatically."
    )
    st.stop()

# ============ HOME PAGE ============
if page == "Home":
    st.header("Welcome!")
    st.markdown("""
    ### What This App Does:
    - 📊 Loads the diabetes dataset
    - 🎯 Trains multiple ML classification models
    - 📈 Compares model performance
    - 🔮 Predicts diabetes outcome for new patients

    👉 Start with **Data Analysis**, then go to **Model Training**,
    and finally use **Make Prediction**.
    """)
    st.info(f"Dataset currently loaded: **{df.shape[0]} rows, {df.shape[1]} columns**")

# ============ DATA ANALYSIS PAGE ============
elif page == "Data Analysis":
    st.header("📊 Data Analysis")

    st.success(f"✅ Dataset loaded: {df.shape[0]} rows")

    if st.checkbox("Show dataset"):
        st.dataframe(df.head(10))

    st.subheader("Statistics")
    st.dataframe(df.describe())

    st.subheader("Outcome Distribution")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        df["Outcome"].value_counts().plot(kind="bar", ax=ax, color=["#4C72B0", "#DD8452"])
        ax.set_xticklabels(["Not Diabetic (0)", "Diabetic (1)"], rotation=0)
        ax.set_ylabel("Count")
        st.pyplot(fig)
    with col2:
        st.write(df["Outcome"].value_counts())

    st.subheader("Feature Correlation")
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm", ax=ax2)
    st.pyplot(fig2)

# ============ MODEL TRAINING PAGE ============
elif page == "Model Training":
    st.header("🤖 Model Training")

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    # Encode any categorical columns (Pima dataset usually has none, but this
    # keeps the app generic/robust if you swap in a different CSV)
    categorical_cols = X.select_dtypes(include="object").columns.tolist()
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    feature_columns = X_encoded.columns.tolist()  # remember for prediction page

    test_size = st.slider("Test set size", 0.1, 0.4, 0.2, 0.05)

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=test_size, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if st.button("🚀 Train Models", use_container_width=True):
        with st.spinner("Training models..."):
            # Logistic Regression (replaces SGDRegressor — this is a
            # classification problem since Outcome is 0/1)
            log_reg = LogisticRegression(max_iter=1000, random_state=42)
            log_reg.fit(X_train_scaled, y_train)
            log_acc = accuracy_score(y_test, log_reg.predict(X_test_scaled))

            # Random Forest Classifier
            rf = RandomForestClassifier(n_estimators=200, random_state=42)
            rf.fit(X_train_scaled, y_train)
            rf_acc = accuracy_score(y_test, rf.predict(X_test_scaled))

            # XGBoost Classifier
            xgb_model = xgb.XGBClassifier(
                n_estimators=200, random_state=42, eval_metric="logloss"
            )
            xgb_model.fit(X_train_scaled, y_train)
            xgb_acc = accuracy_score(y_test, xgb_model.predict(X_test_scaled))

        st.success("✅ Models trained!")

        results = pd.DataFrame({
            "Model": ["Logistic Regression", "Random Forest", "XGBoost"],
            "Accuracy": [log_acc, rf_acc, xgb_acc],
        }).sort_values("Accuracy", ascending=False).reset_index(drop=True)

        st.dataframe(results, use_container_width=True)

        best_row = results.iloc[0]
        st.info(f"🏆 Best model: **{best_row['Model']}** (Accuracy: {best_row['Accuracy']:.3f})")

        models = {
            "Logistic Regression": log_reg,
            "Random Forest": rf,
            "XGBoost": xgb_model,
        }

        # Persist everything the Prediction page needs across reruns
        st.session_state["trained_models"] = models
        st.session_state["scaler"] = scaler
        st.session_state["feature_columns"] = feature_columns
        st.session_state["results"] = results

    elif "results" in st.session_state:
        st.info("Showing results from the last training run:")
        st.dataframe(st.session_state["results"], use_container_width=True)

# ============ PREDICTION PAGE ============
elif page == "Make Prediction":
    st.header("🔮 Make Predictions")

    if "trained_models" not in st.session_state:
        st.warning("⚠️ No trained model found yet. Please go to **Model Training** and click 'Train Models' first.")
        st.stop()

    model_name = st.selectbox("Choose model", list(st.session_state["trained_models"].keys()))

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=17, value=0)
    with col2:
        glucose = st.number_input("Glucose", min_value=0, max_value=300, value=120)
    with col3:
        bp = st.number_input("Blood Pressure", min_value=0, max_value=200, value=80)
    with col4:
        skin = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        insulin = st.number_input("Insulin", min_value=0, max_value=1000, value=79)
    with col6:
        bmi = st.number_input("BMI", min_value=0.0, max_value=60.0, value=32.0)
    with col7:
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
    with col8:
        age = st.number_input("Age", min_value=0, max_value=120, value=33)

    if st.button("🔍 Predict", use_container_width=True):
        input_df = pd.DataFrame([{
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": bp,
            "SkinThickness": skin,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": dpf,
            "Age": age,
        }])

        # Align columns with what the model was trained on
        feature_columns = st.session_state["feature_columns"]
        input_df = input_df.reindex(columns=feature_columns, fill_value=0)

        scaler = st.session_state["scaler"]
        model = st.session_state["trained_models"][model_name]

        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]

        # Probability, if the model supports it (all three here do)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_scaled)[0][1]
        else:
            proba = None

        if prediction == 1:
            st.error(f"⚠️ Prediction: **Diabetic**" + (f" (probability: {proba:.1%})" if proba is not None else ""))
        else:
            st.success(f"✅ Prediction: **Not Diabetic**" + (f" (probability of diabetes: {proba:.1%})" if proba is not None else ""))

# ============ FOOTER ============
st.markdown("---")
st.markdown("<div style='text-align: center;'>Built with Streamlit 🚀</div>", unsafe_allow_html=True)
