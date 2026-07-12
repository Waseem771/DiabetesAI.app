import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# ============ PAGE CONFIGURATION ============
st.set_page_config(page_title="Diabetes Prediction", page_icon="🏥", layout="wide")

st.title("🏥 Diabetes Prediction Model")
st.markdown("---")

# ============ DATA PATH (RELATIVE — WORKS ON STREAMLIT CLOUD) ============
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(APP_DIR, "data", "diabetes_dataset.csv")

# Columns most likely to be the prediction target, in priority order
TARGET_CANDIDATES = [
    "diagnosed_diabetes", "Outcome", "diabetic", "Diabetic", "target", "Diagnosis"
]

# Columns that are usually *derived from* the diagnosis itself — including
# them as model inputs would leak the answer and make accuracy misleadingly
# high. We exclude them by default but let the user override.
LIKELY_LEAKAGE_COLUMNS = ["diabetes_risk_score", "diabetes_stage"]


@st.cache_data
def load_data(path):
    """Load the dataset. Cached so it only runs once."""
    if not os.path.exists(path):
        return None, "not_found"
    try:
        df = pd.read_csv(path)
    except Exception:
        return None, "parse_error"

    if df.shape[1] == 1:
        try:
            df = pd.read_csv(path, sep=None, engine="python")
        except Exception:
            pass

    df.columns = [c.strip() for c in df.columns]
    return df, "ok"


def show_debug_info(path):
    """Self-diagnosing helper for missing/corrupt dataset issues."""
    with st.expander("🔧 Debug info (click to expand)"):
        st.write("**Expected path:**", path)
        st.write("**App directory:**", APP_DIR)
        try:
            st.code("\n".join(sorted(os.listdir(APP_DIR))))
        except Exception as e:
            st.write(f"Could not list app directory: {e}")

        data_dir = os.path.join(APP_DIR, "data")
        if os.path.isdir(data_dir):
            entries = sorted(os.listdir(data_dir))
            st.code("\n".join(entries) if entries else "(empty folder)")
            csv_path = os.path.join(data_dir, "diabetes_dataset.csv")
            if os.path.exists(csv_path):
                size_kb = os.path.getsize(csv_path) / 1024
                st.write(f"**diabetes_dataset.csv size:** {size_kb:.1f} KB")
                if size_kb < 5:
                    st.warning(
                        "⚠️ Suspiciously small for a large dataset — likely a "
                        "Git LFS pointer file rather than the real CSV."
                    )
        else:
            st.code("(no 'data' folder found here)")


def normalize_target(y_raw):
    """
    Convert a target column of unknown type (bool, Yes/No, True/False,
    0/1, or arbitrary categories) into clean 0/1 integer labels.
    Returns (y, label_map) where label_map explains what 1 means.
    """
    if y_raw.dtype == bool:
        return y_raw.astype(int), {0: "False", 1: "True"}

    if pd.api.types.is_numeric_dtype(y_raw):
        uniques = sorted(y_raw.dropna().unique().tolist())
        if set(uniques) <= {0, 1}:
            return y_raw.astype(int), {0: "0", 1: "1"}
        # Numeric but not already 0/1 (e.g. a risk score) — not ideal as a
        # classification target, but binarize around the median as a fallback.
        median = y_raw.median()
        return (y_raw > median).astype(int), {0: f"<= {median}", 1: f"> {median}"}

    # String/object column — map common yes/no style values
    lower_map = {str(v).strip().lower(): v for v in y_raw.dropna().unique()}
    positive_words = {"yes", "true", "diabetic", "1", "positive", "diagnosed"}
    negative_words = {"no", "false", "non-diabetic", "nondiabetic", "0", "negative"}

    if set(lower_map.keys()) <= (positive_words | negative_words):
        y = y_raw.astype(str).str.strip().str.lower().map(
            lambda v: 1 if v in positive_words else 0
        )
        return y, {0: "negative", 1: "positive"}

    # Fallback: generic label encoding (works for any category set, but
    # only makes sense for genuinely binary targets)
    codes, uniques = pd.factorize(y_raw)
    label_map = {i: str(u) for i, u in enumerate(uniques)}
    return pd.Series(codes, index=y_raw.index), label_map


# ============ LOAD DATA ============
df, load_status = load_data(DATASET_PATH)

# ============ SIDEBAR NAVIGATION ============
page = st.sidebar.radio("Select Page:", ["Home", "Data Analysis", "Model Training", "Make Prediction"])

if df is None:
    if load_status == "parse_error":
        st.error(
            "⚠️ Found `data/diabetes_dataset.csv` but couldn't parse it as a CSV. "
            "It may be corrupted or an unresolved Git LFS pointer file."
        )
    else:
        st.error(
            "⚠️ Dataset not found at `data/diabetes_dataset.csv`. Check that it's "
            "committed to your repo at that exact path."
        )
    show_debug_info(DATASET_PATH)
    st.stop()

# ============ HOME PAGE ============
if page == "Home":
    st.header("Welcome!")
    st.markdown("""
    ### What This App Does:
    - 📊 Loads your diabetes risk dataset
    - 🎯 Trains multiple ML classification models
    - 📈 Compares model performance
    - 🔮 Predicts diabetes diagnosis for a new patient profile

    👉 Start with **Data Analysis**, then **Model Training**, then **Make Prediction**.
    """)
    st.info(f"Dataset currently loaded: **{df.shape[0]} rows, {df.shape[1]} columns**")
    st.write("**Columns:**")
    st.code(", ".join(df.columns))

# ============ DATA ANALYSIS PAGE ============
elif page == "Data Analysis":
    st.header("📊 Data Analysis")
    st.success(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    if st.checkbox("Show dataset"):
        st.dataframe(df.head(10))

    st.subheader("Statistics")
    st.dataframe(df.describe(include="all"))

    target_guess = next((c for c in TARGET_CANDIDATES if c in df.columns), None)
    if target_guess:
        st.subheader(f"Target Distribution: `{target_guess}`")
        st.write(df[target_guess].value_counts())

    st.subheader("Numeric Feature Correlation")
    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(numeric_df.corr(), cmap="coolwarm", center=0, ax=ax)
        st.pyplot(fig)
    else:
        st.info("Not enough numeric columns for a correlation heatmap.")

# ============ MODEL TRAINING PAGE ============
elif page == "Model Training":
    st.header("🤖 Model Training")

    default_target = next((c for c in TARGET_CANDIDATES if c in df.columns), df.columns[-1])
    target_col = st.selectbox(
        "Target column (what to predict)",
        options=list(df.columns),
        index=list(df.columns).index(default_target),
    )

    leakage_present = [c for c in LIKELY_LEAKAGE_COLUMNS if c in df.columns and c != target_col]
    exclude_leakage = True
    if leakage_present:
        st.warning(
            f"⚠️ These columns look like they may be **derived from the diagnosis** "
            f"(data leakage): `{', '.join(leakage_present)}`. Including them will "
            f"make accuracy look unrealistically high."
        )
        exclude_leakage = st.checkbox("Exclude these from training (recommended)", value=True)

    exclude_cols = [target_col] + (leakage_present if exclude_leakage else [])
    X = df.drop(columns=exclude_cols)
    y_raw = df[target_col]
    y, label_map = normalize_target(y_raw)

    st.caption(f"Target normalized to: {label_map}")

    categorical_cols = X.select_dtypes(include="object").columns.tolist()
    numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
    st.write(f"**Numerical features ({len(numerical_cols)}):** {', '.join(numerical_cols) or 'none'}")
    st.write(f"**Categorical features ({len(categorical_cols)}):** {', '.join(categorical_cols) or 'none'}")

    test_size = st.slider("Test set size", 0.1, 0.4, 0.2, 0.05)

    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ])

    if st.button("🚀 Train Models", use_container_width=True):
        with st.spinner("Training models..."):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )

            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
                "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1),
                "XGBoost": xgb.XGBClassifier(n_estimators=150, random_state=42, eval_metric="logloss", n_jobs=-1),
            }

            trained_pipelines = {}
            accuracies = {}
            for name, model in models.items():
                pipe = Pipeline([("preprocess", preprocessor), ("model", model)])
                pipe.fit(X_train, y_train)
                acc = accuracy_score(y_test, pipe.predict(X_test))
                trained_pipelines[name] = pipe
                accuracies[name] = acc

        st.success("✅ Models trained!")

        results = pd.DataFrame({
            "Model": list(accuracies.keys()),
            "Accuracy": list(accuracies.values()),
        }).sort_values("Accuracy", ascending=False).reset_index(drop=True)
        st.dataframe(results, use_container_width=True)
        st.info(f"🏆 Best model: **{results.iloc[0]['Model']}** (Accuracy: {results.iloc[0]['Accuracy']:.3f})")

        st.session_state["trained_pipelines"] = trained_pipelines
        st.session_state["results"] = results
        st.session_state["feature_columns"] = X.columns.tolist()
        st.session_state["categorical_cols"] = categorical_cols
        st.session_state["numerical_cols"] = numerical_cols
        st.session_state["label_map"] = label_map
        st.session_state["X_reference"] = X  # for building prediction widgets

    elif "results" in st.session_state:
        st.info("Showing results from the last training run:")
        st.dataframe(st.session_state["results"], use_container_width=True)

# ============ PREDICTION PAGE ============
elif page == "Make Prediction":
    st.header("🔮 Make Predictions")

    if "trained_pipelines" not in st.session_state:
        st.warning("⚠️ No trained model found yet. Please go to **Model Training** and click 'Train Models' first.")
        st.stop()

    model_name = st.selectbox("Choose model", list(st.session_state["trained_pipelines"].keys()))

    X_ref = st.session_state["X_reference"]
    categorical_cols = st.session_state["categorical_cols"]
    numerical_cols = st.session_state["numerical_cols"]
    label_map = st.session_state["label_map"]

    st.caption("Fill in patient details below. Fields are generated from your dataset's actual columns.")

    inputs = {}
    all_cols = st.session_state["feature_columns"]
    cols_per_row = 4
    for i in range(0, len(all_cols), cols_per_row):
        row_cols = st.columns(cols_per_row)
        for j, col_name in enumerate(all_cols[i:i + cols_per_row]):
            with row_cols[j]:
                if col_name in categorical_cols:
                    options = sorted(X_ref[col_name].dropna().unique().tolist())
                    inputs[col_name] = st.selectbox(col_name, options)
                else:
                    col_data = X_ref[col_name].dropna()
                    default_val = float(col_data.median()) if len(col_data) else 0.0
                    inputs[col_name] = st.number_input(
                        col_name,
                        value=default_val,
                        format="%.3f",
                    )

    if st.button("🔍 Predict", use_container_width=True):
        input_df = pd.DataFrame([inputs])[all_cols]
        pipe = st.session_state["trained_pipelines"][model_name]

        prediction = pipe.predict(input_df)[0]
        proba = None
        if hasattr(pipe, "predict_proba"):
            proba = pipe.predict_proba(input_df)[0][1]

        result_label = label_map.get(prediction, str(prediction))
        if prediction == 1:
            st.error(f"⚠️ Prediction: **{result_label}**" + (f" (probability: {proba:.1%})" if proba is not None else ""))
        else:
            st.success(f"✅ Prediction: **{result_label}**" + (f" (probability of positive class: {proba:.1%})" if proba is not None else ""))

# ============ FOOTER ============
st.markdown("---")
st.markdown("<div style='text-align: center;'>Built with Streamlit 🚀</div>", unsafe_allow_html=True)
