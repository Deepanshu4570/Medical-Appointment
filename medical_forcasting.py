import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    f1_score, roc_auc_score, confusion_matrix,
    roc_curve, precision_recall_curve,
    mean_absolute_error, mean_squared_error
)

# ── SETUP ────────────────────────────────────────────────────
st.set_page_config(page_title="Medical Dashboard", page_icon="🏥", layout="wide")
st.title("🏥 Medical Appointment Intelligence Dashboard")

BASE = os.path.dirname(os.path.abspath(__file__))

# ── LOAD MODELS ──────────────────────────────────────────────
try:
    classifier = joblib.load(os.path.join(BASE, "noshow_classifier_final.joblib"))
    forecaster = joblib.load(os.path.join(BASE, "demand_forecast.joblib"))
    st.sidebar.success("✅ Models loaded")
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

# ── FEATURES ─────────────────────────────────────────────────
FEATURES = [
    'specialty', 'appointment_time', 'appointment_shift', 'age',
    'patient_needs_companion', 'SMS_received',
    'day_of_week', 'is_weekend', 'health_score',
    'age_health_interaction', 'weather_stress',
    'waiting_days', 'bad_weather', 'high_risk_patient',
    'daily_load', 'place_freq'
]

# ── NAVIGATION ───────────────────────────────────────────────
page = st.sidebar.radio("Navigate", [
    "📊 Overview",
    "🤖 No-Show Model",
    "📉 Forecast",
    "🧪 Live Predict"
])

# ── DATA LOAD ────────────────────────────────────────────────
if page != "🧪 Live Predict":
    csv = st.sidebar.file_uploader("Upload CSV", type="csv")
    if csv is None:
        st.info("Upload dataset to continue")
        st.stop()

    df = pd.read_csv(csv)

    # ── PREPROCESSING ────────────────────────────────────────
    df['appointment_date'] = pd.to_datetime(df['appointment_date_continuous'], errors='coerce')
    df['scheduled_day'] = pd.to_datetime(df.get('scheduled_day'), errors='coerce')

    df['no_show'] = df['no_show'].map({'no': 0, 'yes': 1})
    df['age'] = df['age'].fillna(df['age'].median())

    df['day_of_week'] = df['appointment_date'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)

    df['health_score'] = (
        df.get('Hipertension',0) +
        df.get('Diabetes',0) +
        df.get('Alcoholism',0) +
        df.get('Handcap',0)
    )

    df['age_health_interaction'] = df['age'] * df['health_score']

    df['weather_stress'] = df.get('average_temp_day',0) * df.get('average_rain_day',0)

    df['waiting_days'] = (df['appointment_date'] - df['scheduled_day']).dt.days.fillna(0)

    df['bad_weather'] = (
        (df.get('rainy_day_before',0)==1) |
        (df.get('storm_day_before',0)==1)
    ).astype(int)

    df['high_risk_patient'] = ((df['age'] < 25) & (df['health_score']==0)).astype(int)

    df['daily_load'] = df.groupby('appointment_date')['appointment_date'].transform('count')

    df['place_freq'] = (
        df['place'].map(df['place'].value_counts(normalize=True))
        if 'place' in df.columns else 0
    )

    for col in ['specialty','appointment_shift']:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    df = df.fillna(0)

# ════════════════════════════════════
# 📊 OVERVIEW
# ════════════════════════════════════
if page == "📊 Overview":
    st.metric("Total Records", len(df))
    st.metric("No-Show Rate", f"{df['no_show'].mean()*100:.2f}%")

    st.dataframe(df.head())

    st.subheader("Daily Demand")
    st.line_chart(df.groupby('appointment_date').size())

# ════════════════════════════════════
# 🤖 MODEL + SHAP
# ════════════════════════════════════
elif page == "🤖 No-Show Model":

    X = df[FEATURES]
    y = df['no_show']

    y_pred = classifier.predict(X)
    y_prob = classifier.predict_proba(X)[:,1]

    c1,c2,c3 = st.columns(3)
    c1.metric("F1", f"{f1_score(y,y_pred):.3f}")
    c2.metric("ROC-AUC", f"{roc_auc_score(y,y_prob):.3f}")
    c3.metric("Avg Risk", f"{y_prob.mean()*100:.1f}%")

    # Confusion matrix
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y,y_pred), annot=True, fmt='d', cmap='Blues')
    st.pyplot(fig); plt.close()

    # ROC
    fpr,tpr,_ = roc_curve(y,y_prob)
    fig, ax = plt.subplots()
    ax.plot(fpr,tpr)
    ax.plot([0,1],[0,1],'--')
    st.pyplot(fig); plt.close()

    # SHAP
    st.subheader("🔍 SHAP Explainability")

    X_sample = X.sample(min(500,len(X)), random_state=42)

    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    expected_value = explainer.expected_value
    if isinstance(expected_value, list):
        expected_value = expected_value[1]

    # Global
    fig = plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False)
    st.pyplot(fig); plt.close()

    # Bar
    fig = plt.figure()
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    st.pyplot(fig); plt.close()

    # Individual
    st.subheader("Explain Individual Prediction")
    idx = st.slider("Select row", 0, len(X_sample)-1, 0)

    fig = plt.figure()
    shap.force_plot(
        expected_value,
        shap_values[idx],
        X_sample.iloc[idx],
        matplotlib=True,
        show=False
    )
    st.pyplot(fig); plt.close()

# ════════════════════════════════════
# 📉 FORECAST
# ════════════════════════════════════
elif page == "📉 Forecast":

    d = df.groupby('appointment_date').size().reset_index(name='y')
    d.columns = ['ds','y']
    d = d.sort_values('ds')

    d['dayofweek'] = d['ds'].dt.dayofweek
    d['month'] = d['ds'].dt.month
    d['lag_1'] = d['y'].shift(1)
    d['lag_2'] = d['y'].shift(2)
    d['lag_7'] = d['y'].shift(7)
    d['rolling_7'] = d['y'].shift(1).rolling(7).mean()
    d['trend'] = np.arange(len(d))

    d = d.dropna()

    FCOLS = ['dayofweek','month','lag_1','lag_2','lag_7','rolling_7','trend']

    pred = np.clip(np.expm1(forecaster.predict(d[FCOLS])),0,None)

    mae = mean_absolute_error(d['y'], pred)
    rmse = np.sqrt(mean_squared_error(d['y'], pred))

    st.metric("MAE", f"{mae:.2f}")
    st.metric("RMSE", f"{rmse:.2f}")

    fig, ax = plt.subplots()
    ax.plot(d['ds'], d['y'], label="Actual")
    ax.plot(d['ds'], pred, label="Predicted")
    ax.legend()
    st.pyplot(fig); plt.close()

    # Residuals
    res = d['y'] - pred
    fig, ax = plt.subplots()
    sns.histplot(res, bins=30)
    st.pyplot(fig); plt.close()

    # Future
    st.subheader("Next 7 Days Forecast")

    last = d.iloc[-1:].copy()
    future = []

    for _ in range(7):
        new = last.copy()
        new['ds'] += pd.Timedelta(days=1)
        new['trend'] += 1
        new['dayofweek'] = new['ds'].dt.dayofweek
        new['month'] = new['ds'].dt.month

        yhat = np.clip(np.expm1(forecaster.predict(new[FCOLS])),0,None)[0]
        new['y'] = yhat

        future.append(new)
        last = new

    future_df = pd.concat(future)
    st.dataframe(future_df[['ds','y']])

# ════════════════════════════════════
# 🧪 LIVE PREDICT + SHAP
# ════════════════════════════════════
elif page == "🧪 Live Predict":

    st.subheader("Predict + Explain")

    age = st.slider("Age",0,100,35)
    sms = st.selectbox("SMS", [1,0])
    companion = st.selectbox("Companion", [0,1])
    health_score = st.slider("Health Score",0,4,1)
    waiting_days = st.slider("Waiting Days",0,60,5)

    specialty = st.number_input("Specialty",0,30,5)
    shift = st.number_input("Shift",0,5,1)
    time = st.slider("Hour",0,23,10)
    weather = st.slider("Weather Stress",0.0,500.0,50.0)
    place_freq = st.slider("Place Freq",0.0,1.0,0.05)

    day = st.slider("Day of Week",0,6,2)
    weekend = int(day in [5,6])
    bad_weather = st.selectbox("Bad Weather", [0,1])
    high_risk = int((age<25) and (health_score==0))
    daily_load = st.slider("Daily Load",0,500,100)

    if st.button("Predict"):

        row = pd.DataFrame([[specialty,time,shift,age,companion,sms,
                             day,weekend,health_score,
                             age*health_score,weather,
                             waiting_days,bad_weather,high_risk,
                             daily_load,place_freq]],
                           columns=FEATURES)

        prob = classifier.predict_proba(row)[0][1]

        st.success(f"No-Show Probability: {prob:.2%}")

        # SHAP
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(row)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        expected_value = explainer.expected_value
        if isinstance(expected_value, list):
            expected_value = expected_value[1]

        st.subheader("Why this prediction?")

        fig = plt.figure()
        shap.force_plot(
            expected_value,
            shap_values[0],
            row.iloc[0],
            matplotlib=True,
            show=False
        )
        st.pyplot(fig); plt.close()