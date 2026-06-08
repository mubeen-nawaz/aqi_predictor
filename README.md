# 🌿 Pearls AQI Predictor

> Predict the Air Quality Index for the next 3 days — 100% free, no Hopsworks needed!

**By: Muhammad Mubeen Nawaz | Data Science | CGPA: 3.92**

---

## 📌 Project Overview

This project predicts Air Quality Index (AQI) for the next 3 days using:
- Real-time weather data (OpenWeather API — free tier)
- Real AQI data (AQICN API — free tier)
- Multiple ML models (Random Forest, Gradient Boosting, Ridge)
- Beautiful Streamlit web dashboard
- Automated daily retraining via GitHub Actions

**No Hopsworks, no paid services needed!**

---

## 🏗️ Technology Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11 |
| ML Models | scikit-learn (Random Forest, Gradient Boosting, Ridge) |
| Feature Store | Local CSV (Instead of Hopsworks) |
| Model Registry | Local `models/` folder |
| Dashboard | Streamlit + Plotly |
| Data APIs | OpenWeather API + AQICN API (both free) |
| CI/CD | GitHub Actions (free) |
| Explainability | Permutation Importance (SHAP-style) |

---

## 📁 Project Structure

```
aqi-predictor/
├── src/
│   ├── features.py      # Data fetch + feature engineering
│   ├── train.py         # ML model training pipeline
│   ├── predict.py       # 3-day AQI forecast
│   └── scheduler.py     # Automated hourly data collection
├── dashboard/
│   └── app.py           # Streamlit web dashboard
├── tests/
│   └── test_pipeline.py # Unit tests
├── data/
│   └── feature_store.csv  # Local feature store (auto-generated)
├── models/
│   ├── best_model.joblib  # Trained model (auto-generated)
│   ├── scaler.joblib      # Feature scaler
│   └── metadata.json      # Model metrics & feature importance
├── .github/
│   └── workflows/
│       └── daily_train.yml  # GitHub Actions CI/CD
├── .env.example         # Environment variables template
├── .gitignore           # Protects secrets
└── requirements.txt     # Python dependencies
```

---

## 🚀 Setup Instructions (Step by Step)

### Step 1: Download the Project
```bash
git clone https://github.com/YOUR_USERNAME/aqi-predictor.git
cd aqi-predictor
```

### Step 2: Create a Python Environment
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Mac/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Get Free API Keys

**OpenWeather API (Free):**
1. Go to https://openweathermap.org/api
2. "Sign Up" (free)
3. Go to the "API Keys" section
4. Copy the key

**AQICN API (Free):**
1. Go to https://aqicn.org/api/
2. Enter your email to get a free key

### Step 5: Create a .env file
```bash
# Copy .env.example
cp .env.example .env

# Add your keys to the .env file:
OPENWEATHER_API_KEY=abcd1234...   ← your real key here
AQICN_API_KEY=xyz789...           ← your real key here
CITY_NAME=Lahore
COUNTRY_CODE=PK
```

### Step 6: Fetch data for the first time
```bash
python src/features.py
```

### Step 7: Train the model
```bash
python src/train.py
```

### Step 8: Run the dashboard
```bash
streamlit run dashboard/app.py
```
It will open in the browser at: http://localhost:8501

---

## 🤖 Automated Pipeline (Optional)

Run in the background - hourly data fetch, daily training:
```bash
python src/scheduler.py
```

---

## 📊 Key Features

### ✅ Feature Engineering
- Real-time weather (temperature, humidity, pressure, wind)
- Time-based features (hour, day, month, weekend)
- Derived features (heat index, wind chill, calm/humid flags)
- AQI change rate

### ✅ Historical Data Backfill
- 500+ synthetic training records generated automatically
- Real data accumulated over time via scheduler

### ✅ ML Models Compared
- Random Forest (usually wins)
- Gradient Boosting
- Ridge Regression
- Best model auto-selected by MAE

### ✅ Model Evaluation
- MAE, RMSE, R² metrics
- Cross-validation
- Permutation feature importance (SHAP-style)

### ✅ 3-Day Forecast Dashboard
- Hourly AQI predictions
- Daily summary cards
- Health recommendations in English
- Historical data chart

### ✅ Automated CI/CD (GitHub Actions)
- Runs daily at 2am Pakistan time
- Fetches fresh data
- Retrains model
- Saves artifacts

---

## 🔐 How to Add Keys to GitHub (Secrets)

**⚠️ NEVER push the .env file to GitHub!**

Use GitHub Secrets:
1. Go to your GitHub repo
2. **Settings** → **Secrets and variables** → **Actions**
3. Click **"New repository secret"**
4. Create these secrets:
   - Name: `OPENWEATHER_API_KEY` | Value: your key
   - Name: `AQICN_API_KEY` | Value: your key
   - Name: `CITY_NAME` | Value: `Lahore`

Now GitHub Actions will automatically use these keys — safe!

---

## 🧪 Run Tests
```bash
pytest tests/ -v
```

---

## 📈 Model Performance (Typical)

| Model | MAE | RMSE | R² |
|-------|-----|------|-----|
| Random Forest | ~12 | ~18 | ~0.85 |
| Gradient Boosting | ~14 | ~20 | ~0.82 |
| Ridge Regression | ~22 | ~30 | ~0.65 |

*(Actual results depend on real data collected)*

---

## 🎯 AQI Categories

| AQI Range | Category | Health Impact |
|-----------|----------|---------------|
| 0-50 | 🟢 Good | Safe for outdoor activities |
| 51-100 | 🟡 Moderate | Precautions for sensitive individuals |
| 101-150 | 🟠 Unhealthy for Sensitive | Precautions for children and elderly |
| 151-200 | 🔴 Unhealthy | Precautions for everyone, wear a mask |
| 201-300 | 🟣 Very Unhealthy | Stay indoors |
| 301+ | ⛔ Hazardous | Do not go outside at all |

---

## 👩‍💻 Author

**Muhammad Mubeen Nawaz**
- University: Other (The Islamia University of Bahawalpur)
- Domain: Data Science
- CGPA: 3.92

---

*Pearls AQI Predictor — Predicting clean air, one day at a time 🌿*
