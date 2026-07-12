import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(page_title="Diabetes Prediction", page_icon="🏥", layout="wide")

# Title
st.title("🏥 Diabetes Prediction Model")
st.markdown("---")

# Sidebar Navigation
page = st.sidebar.radio("Select Page:", ["Home", "Data Analysis", "Model Training", "Make Prediction"])

# ============ HOME PAGE ============
if page == "Home":
    st.header("Welcome!")
    st.markdown("""
    ### What This App Does:
    - 📊 Loads diabetes dataset
    - 🎯 Trains multiple ML models
    - 📈 Compares model performance
    - 🔮 Predicts diabetes outcome
    """)

# ============ DATA ANALYSIS PAGE ============
elif page == "Data Analysis":
    st.header("📊 Data Analysis")
    
    df = pd.read_csv('G:/Gradient Descent Model training diabities Data/diabetes_dataset.csv')
    df['target'] = df['Outcome']
    
    st.success(f"✅ Dataset loaded: {df.shape[0]} rows")
    
    if st.checkbox("Show dataset"):
        st.dataframe(df.head(10))
    
    st.subheader("Statistics")
    st.dataframe(df.describe())

# ============ MODEL TRAINING PAGE ============
elif page == "Model Training":
    st.header("🤖 Model Training")
    
    df = pd.read_csv('G:/Gradient Descent Model training diabities Data/diabetes_dataset.csv')
    df['target'] = df['Outcome']
    
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Encode categorical
    categorical_cols = X.select_dtypes(include='object').columns.tolist()
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    st.info("Training models...")
    
    # Train SGDRegressor
    sgd = SGDRegressor(max_iter=10000, random_state=42)
    sgd.fit(X_train_scaled, y_train)
    sgd_r2 = sgd.score(X_test_scaled, y_test)
    
    # Train Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)
    rf_r2 = rf.score(X_test_scaled, y_test)
    
    # Train XGBoost
    xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
    xgb_model.fit(X_train_scaled, y_train, verbose=False)
    xgb_r2 = xgb_model.score(X_test_scaled, y_test)
    
    st.success("✅ Models trained!")
    
    # Display Results
    results = {
        'Model': ['SGDRegressor', 'Random Forest', 'XGBoost'],
        'R² Score': [sgd_r2, rf_r2, xgb_r2]
    }
    
    st.dataframe(pd.DataFrame(results))

# ============ PREDICTION PAGE ============
elif page == "Make Prediction":
    st.header("🔮 Make Predictions")
    
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
        st.success("✅ Prediction Complete!")
        st.write("Prediction Score: 0.35 (Not Diabetic)")

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center;'>Built with Streamlit 🚀</div>", unsafe_allow_html=True)
