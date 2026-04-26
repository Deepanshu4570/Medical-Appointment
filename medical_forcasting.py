import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ── SETUP ────────────────────────────────────────────────────
st.set_page_config(page_title="Medical Dashboard", page_icon="🏥", layout="wide")
st.title("🏥 Medical Appointment Dashboard")

# ── LOAD MODELS ──────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

try:
    classifier = joblib.load(os.path.join(BASE, "noshow_classifier_final.joblib"))
    forecaster = joblib.load(os.path.join(BASE, "demand_forecast.joblib"))
    st.sidebar.success("✅ Models loaded!")
except Exception as e:
    st.error(f"❌ Could not load model files.\n\n{e}")
    st.stop()

# ── NAVIGATION ───────────────────────────────────────────────
page = st.sidebar.radio("Go to", ["📊 Overview", "🤖 No-Show Model", "📉 Forecast", "🧪 Live Predict"])

# ── UPLOAD CSV (not needed for Live Predict) ─────────────────
FEATURES = ['specialty','appointment_time','appointment_shift','age',
            'patient_needs_companion','SMS_received',
            'weather_stress','bad_weather','high_risk_patient','place_freq']

if page != "🧪 Live Predict":
    csv = st.sidebar.file_uploader("Upload CSV", type="csv")
    if csv is None:
        st.info("👈 Please upload your CSV file from the sidebar.")
        st.stop()

    # Basic preprocessing
    from sklearn.preprocessing import LabelEncoder
    df = pd.read_csv(csv)
    df['appointment_date'] = pd.to_datetime(df['appointment_date_continuous'], errors='coerce')
    df['no_show']          = df['no_show'].map({'no': 0, 'yes': 1})
    df['month']            = df['appointment_date'].dt.month
    df['age']              = df['age'].fillna(df['age'].median())
    df['health_score']     = df.get('Hipertension',0) + df.get('Diabetes',0) + df.get('Alcoholism',0) + df.get('Handcap',0)
    df['bad_weather']      = ((df.get('rainy_day_before',0)==1) | (df.get('storm_day_before',0)==1)).astype(int)
    df['high_risk_patient']= ((df['age'] < 25) & (df['health_score']==0)).astype(int)
    df['weather_stress']   = df.get('average_temp_day',0) * df.get('average_rain_day',0)
    df['place_freq']       = df['place'].map(df['place'].value_counts(normalize=True)) if 'place' in df.columns else 0
    for col in ['specialty','appointment_shift','gender','disability']:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))


# ════════════════════════════════════
# PAGE 1 — OVERVIEW
# ════════════════════════════════════
if page == "📊 Overview":

    # Top metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Records", f"{len(df):,}")
    c2.metric("No-Show Rate",  f"{df['no_show'].mean()*100:.1f}%")
    c3.metric("Total Columns",  df.shape[1])

    st.subheader("Sample Data")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Daily Appointment Demand")
    st.line_chart(df.groupby('appointment_date').size())


# ════════════════════════════════════
# PAGE 2 — NO-SHOW MODEL
# ════════════════════════════════════
elif page == "🤖 No-Show Model":
    from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
    import seaborn as sns

    st.subheader("No-Show Classifier — Results")

    X      = df[[c for c in FEATURES if c in df.columns]].fillna(0)
    y      = df['no_show']
    y_pred = classifier.predict(X)
    y_prob = classifier.predict_proba(X)[:,1]

    c1, c2 = st.columns(2)
    c1.metric("F1 Score", f"{f1_score(y, y_pred):.3f}")
    c2.metric("ROC-AUC",  f"{roc_auc_score(y, y_prob):.3f}")

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Show','No-Show'], yticklabels=['Show','No-Show'])
        ax.set_title("Confusion Matrix")
        st.pyplot(fig); plt.close()

    with col2:
        imp = pd.Series(classifier.feature_importances_,
                        index=[c for c in FEATURES if c in df.columns]).sort_values()
        fig, ax = plt.subplots()
        imp.plot.barh(ax=ax, color='steelblue')
        ax.set_title("Feature Importance")
        st.pyplot(fig); plt.close()


# ════════════════════════════════════
# PAGE 3 — DEMAND FORECAST
# ════════════════════════════════════
elif page == "📉 Forecast":
    from sklearn.metrics import mean_absolute_error

    st.subheader("Daily Demand Forecast")

    # Build features
    d = df.groupby('appointment_date').size().reset_index(name='y')
    d.columns = ['ds','y']
    d = d.sort_values('ds').reset_index(drop=True)
    d['dayofweek'] = d['ds'].dt.dayofweek
    d['month']     = d['ds'].dt.month
    d['lag_1']     = d['y'].shift(1)
    d['lag_2']     = d['y'].shift(2)
    d['lag_7']     = d['y'].shift(7)
    d['rolling_7'] = d['y'].shift(1).rolling(7).mean()
    d['trend']     = np.arange(len(d))
    d = d.dropna()

    FCOLS = ['dayofweek','month','lag_1','lag_2','lag_7','rolling_7','trend']
    split = int(len(d) * 0.8)
    train, test = d.iloc[:split], d.iloc[split:]

    pred = np.clip(np.expm1(forecaster.predict(test[FCOLS])), 0, None)
    mae  = mean_absolute_error(test['y'], pred)

    st.metric("MAE", f"{mae:.1f} appointments/day")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(train['ds'], train['y'], label='Train',     color='steelblue')
    ax.plot(test['ds'],  test['y'],  label='Actual',    color='green')
    ax.plot(test['ds'],  pred,       label='Predicted', color='red', linestyle='--')
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_title("Actual vs Predicted Demand")
    st.pyplot(fig); plt.close()


# ════════════════════════════════════
# PAGE 4 — LIVE PREDICT
# ════════════════════════════════════
elif page == "🧪 Live Predict":
    st.subheader("Predict No-Show Risk for One Patient")

    col1, col2 = st.columns(2)
    with col1:
        age         = st.slider("Age", 0, 100, 35)
        sms         = st.selectbox("SMS Sent?",         [1,0], format_func=lambda x: "Yes" if x else "No")
        companion   = st.selectbox("Needs Companion?",  [0,1], format_func=lambda x: "Yes" if x else "No")
        high_risk   = st.selectbox("High Risk?",        [0,1], format_func=lambda x: "Yes" if x else "No")
        bad_weather = st.selectbox("Bad Weather?",      [0,1], format_func=lambda x: "Yes" if x else "No")
    with col2:
        specialty   = st.number_input("Specialty (encoded)",  0, 30, 5)
        appt_shift  = st.number_input("Shift (encoded)",      0,  5, 1)
        appt_time   = st.slider("Appointment Hour",    0, 23, 10)
        weather_str = st.slider("Weather Stress",      0.0, 500.0, 50.0)
        place_freq  = st.slider("Clinic Frequency",    0.0,   1.0,  0.05)

    if st.button("🔍 Predict", type="primary", use_container_width=True):
        row  = pd.DataFrame([[specialty, appt_time, appt_shift, age, companion,
                               sms, weather_str, bad_weather, high_risk, place_freq]],
                             columns=FEATURES)
        prob  = classifier.predict_proba(row)[0][1]
        label = "⚠️ Likely NO-SHOW" if prob > 0.5 else "✅ Likely to SHOW UP"
        color = "#F44336"           if prob > 0.5 else "#4CAF50"
        risk  = "🔴 HIGH" if prob > 0.6 else ("🟡 MEDIUM" if prob > 0.35 else "🟢 LOW")

        st.markdown(f"""
        <div style="background:{color}22; border-left:6px solid {color};
                    padding:1rem; border-radius:10px; text-align:center">
            <h2 style="color:{color}">{label}</h2>
            <h3>Probability: {prob*100:.1f}% &nbsp;|&nbsp; Risk: {risk}</h3>
        </div>""", unsafe_allow_html=True)

        