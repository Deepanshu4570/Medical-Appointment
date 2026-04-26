# 🏥 Medical Appointment No-Show Prediction & Demand Forecasting

A machine learning project that predicts whether a patient will miss their medical appointment and forecasts daily appointment demand — built using real appointment data with weather, patient health, and scheduling features.

---

## 📌 Problem Statement

Two core problems solved:
1. **No-Show Prediction** — Will a patient show up for their appointment? (Binary Classification)
2. **Demand Forecasting** — How many appointments will be booked on a given day? (Time Series Regression)

---

## 📂 Dataset

**File:** `Medical_appointment_data.csv`

Key columns used:

| Column | Description |
|--------|-------------|
| `age` | Patient age |
| `gender` | Patient gender |
| `specialty` | Medical specialty |
| `appointment_date_continuous` | Date of appointment |
| `appointment_time` | Hour of appointment |
| `appointment_shift` | Morning / Evening shift |
| `SMS_received` | Whether patient got an SMS reminder |
| `patient_needs_companion` | Whether patient needs accompaniment |
| `Hipertension`, `Diabetes`, `Alcoholism`, `Handcap` | Patient health conditions |
| `rainy_day_before`, `storm_day_before` | Weather conditions |
| `average_temp_day`, `average_rain_day` | Weather intensity |
| `rain_intensity`, `heat_intensity` | Categorical weather labels |
| `place` | Clinic/Hospital location |
| `disability` | Disability status |
| `no_show` | Target — `yes` / `no` |

---

## 🔧 Feature Engineering

Features created during preprocessing:

- **`health_score`** — Sum of Hipertension + Diabetes + Alcoholism + Handcap
- **`bad_weather`** — 1 if rainy or stormy day before appointment
- **`high_risk_patient`** — 1 if age < 25 AND health_score == 0
- **`age_health_interaction`** — age × health_score
- **`weather_stress`** — average_temp_day × average_rain_day
- **`waiting_days`** — Days from first record to appointment
- **`daily_load`** — Total appointments on that date
- **`place_freq`** — Frequency encoding of clinic location
- **`time_shift_combo`** — Combination of appointment time + shift
- **`is_weekend`** — 1 if Saturday or Sunday
- **`age_group`** — Binned: child / teen / adult / middle / senior

---

## 🤖 Models Used

### Part 1 — No-Show Classification

| Model | Notes |
|-------|-------|
| Logistic Regression | Baseline, `class_weight='balanced'` |
| Random Forest | 150 trees, max_depth=18 |
| Gradient Boosting | Sklearn default |
| **XGBoost** ✅ | Best model — `scale_pos_weight=2.15` |
| LightGBM | 200 estimators, `class_weight='balanced'` |

**Best Model: XGBoost**

- Evaluated using: F1 Score, ROC-AUC, Confusion Matrix
- Class imbalance handled with **SMOTE** oversampling
- Train/Test split: 80/20 with `stratify=y`

**Final features used:**
```
specialty, appointment_time, appointment_shift, age,
patient_needs_companion, SMS_received, weather_stress,
bad_weather, high_risk_patient, place_freq
```

### Part 2 — Demand Forecasting (Regression)

| Model | Notes |
|-------|-------|
| **XGBoost Regressor** ✅ | 500 estimators, lr=0.03 |
| LightGBM Regressor | 500 estimators, lr=0.03 |
| Prophet | Facebook Prophet on raw daily counts |

**Features used for forecasting:**
```
dayofweek, month, lag_1, lag_2, lag_7, rolling_7, trend
```

Target was **log-transformed** (`np.log1p`) before training and inverse-transformed (`np.expm1`) after prediction.

**Evaluation metrics:** MAE, MAPE, R²

---

## 📊 Streamlit Dashboard

Run the interactive dashboard with:

```bash
streamlit run app.py
```

**Pages:**
- 📊 Overview — Dataset stats, missing values, sample data
- 📈 EDA — No-show patterns by day, SMS, age, weather
- 🤖 No-Show Model — Confusion matrix, ROC-AUC, feature importance
- 📉 Demand Forecast — Actual vs predicted demand chart
- 🧪 Live Predict — Enter patient details, get real-time no-show risk

---

## 🚀 How to Run

**1. Clone the repo**
```bash
git clone https://github.com/your-username/medical-forecasting.git
cd medical-forecasting
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the notebook**
```bash
jupyter notebook Medical_forcasting.ipynb
```

**4. Launch the dashboard**
```bash
streamlit run app.py
```

---

## 📦 Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
xgboost
lightgbm
prophet
streamlit
joblib
```

---

## 📁 Project Structure

```
medical-forecasting/
│
├── Medical_forcasting.ipynb      # Main notebook (EDA + Models)
├── app.py                        # Streamlit dashboard
├── Medical_appointment_data.csv  # Dataset (add manually)
├── noshow_classifier_final.joblib # Saved XGBoost model
├── requirements.txt
└── README.md
```

---

## 💡 Key Insights from EDA

- Patients who received **no SMS** had a higher no-show rate
- **Younger patients (age < 25)** with no health conditions were flagged as high risk
- **Bad weather days** correlated with increased no-shows
- Appointment demand showed clear **weekly seasonality**
- Strongest positive correlation found between `daily_load` and appointment volume

---

## 👤 Author

Built as a final-stage ML project covering end-to-end data science: EDA → Feature Engineering → Classification → Forecasting → Deployment.
