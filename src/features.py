"""
Feature Engineering Pipeline
Uses local CSV feature store instead of Hopsworks - 100% free!
"""

import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
import json

load_dotenv()

OPENWEATHER_KEY = os.getenv("OPENWEATHER_API_KEY")
AQICN_KEY       = os.getenv("AQICN_API_KEY")
CITY            = os.getenv("CITY_NAME", "Lahore")
COUNTRY         = os.getenv("COUNTRY_CODE", "PK")

# Local feature store path (instead of Hopsworks)
FEATURE_STORE = "data/feature_store.csv"
os.makedirs("data", exist_ok=True)


def fetch_weather(city: str = CITY) -> dict:
    """Fetch weather data from OpenWeather API (free tier)."""
    if not OPENWEATHER_KEY or OPENWEATHER_KEY == "your_openweather_api_key_here":
        # Demo data if key is not available
        print("⚠️  OpenWeather key not found - using demo data")
        return _demo_weather()

    url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?q={city},{COUNTRY}&appid={OPENWEATHER_KEY}&units=metric"
    )
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        return {
            "temperature": data["main"]["temp"],
            "humidity":    data["main"]["humidity"],
            "pressure":    data["main"]["pressure"],
            "wind_speed":  data["wind"]["speed"],
            "wind_deg":    data["wind"].get("deg", 0),
            "clouds":      data["clouds"]["all"],
            "weather_main": data["weather"][0]["main"],
        }
    except Exception as e:
        print(f"Weather API error: {e} - using demo data")
        return _demo_weather()


def fetch_aqi(city: str = CITY) -> float:
    """Fetch real AQI from AQICN API (free tier)."""
    if not AQICN_KEY or AQICN_KEY == "your_aqicn_api_key_here":
        print("⚠️  AQICN key not found - using demo AQI")
        return float(np.random.randint(80, 200))

    url = f"https://api.waqi.info/feed/{city}/?token={AQICN_KEY}"
    try:
        r = requests.get(url, timeout=10)
        data = r.json()
        if data["status"] == "ok":
            return float(data["data"]["aqi"])
    except Exception as e:
        print(f"AQICN error: {e}")
    return float(np.random.randint(80, 200))


def compute_features(weather: dict, aqi: float, timestamp: datetime = None) -> dict:
    """Compute ML features from raw data."""
    if timestamp is None:
        timestamp = datetime.now()

    features = {
        "timestamp":    timestamp.isoformat(),
        "hour":         timestamp.hour,
        "day_of_week":  timestamp.weekday(),
        "month":        timestamp.month,
        "is_weekend":   int(timestamp.weekday() >= 5),
        # Weather features
        "temperature":  weather.get("temperature", 25.0),
        "humidity":     weather.get("humidity", 60.0),
        "pressure":     weather.get("pressure", 1013.0),
        "wind_speed":   weather.get("wind_speed", 3.0),
        "wind_deg":     weather.get("wind_deg", 180),
        "clouds":       weather.get("clouds", 30),
        # Derived features
        "heat_index":   _heat_index(weather.get("temperature", 25), weather.get("humidity", 60)),
        "wind_chill":   _wind_chill(weather.get("temperature", 25), weather.get("wind_speed", 3)),
        "is_calm":      int(weather.get("wind_speed", 3) < 2),
        "is_humid":     int(weather.get("humidity", 60) > 80),
        # Target
        "aqi":          aqi,
        "aqi_category": _aqi_category(aqi),
    }
    return features


def save_to_feature_store(features: dict):
    """Save to local CSV feature store (instead of Hopsworks)."""
    df_new = pd.DataFrame([features])
    if os.path.exists(FEATURE_STORE):
        df = pd.read_csv(FEATURE_STORE)
        df = pd.concat([df, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(FEATURE_STORE, index=False)
    print(f"✅ Feature store updated: {FEATURE_STORE} ({len(df)} records)")


def load_feature_store() -> pd.DataFrame:
    """Load data from feature store."""
    if not os.path.exists(FEATURE_STORE):
        print("Feature store not found - generating demo data...")
        return _generate_demo_data()
    df = pd.read_csv(FEATURE_STORE)
    if len(df) < 50:
        print(f"Only {len(df)} records found - adding demo data as well...")
        demo = _generate_demo_data()
        df = pd.concat([demo, df], ignore_index=True)
    return df


def run_feature_pipeline():
    """Run complete feature pipeline - run once or schedule it."""
    print(f"🔄 Feature pipeline running - {datetime.now().strftime('%H:%M:%S')}")
    weather = fetch_weather()
    aqi     = fetch_aqi()
    feats   = compute_features(weather, aqi)
    save_to_feature_store(feats)
    print(f"   AQI: {aqi:.0f} | Temp: {weather['temperature']:.1f}°C | Humidity: {weather['humidity']}%")
    return feats


# ── Helper functions ──────────────────────────────────────────────────────────

def _heat_index(temp: float, humidity: float) -> float:
    return temp + 0.33 * (humidity / 100 * 6.105) - 4.0

def _wind_chill(temp: float, wind_speed: float) -> float:
    if wind_speed < 1:
        return temp
    return 13.12 + 0.6215 * temp - 11.37 * (wind_speed ** 0.16) + 0.3965 * temp * (wind_speed ** 0.16)

def _aqi_category(aqi: float) -> str:
    if aqi <= 50:   return "Good"
    if aqi <= 100:  return "Moderate"
    if aqi <= 150:  return "Unhealthy for Sensitive"
    if aqi <= 200:  return "Unhealthy"
    if aqi <= 300:  return "Very Unhealthy"
    return "Hazardous"

def _demo_weather() -> dict:
    """Return demo weather data if API key is not available."""
    return {
        "temperature": round(np.random.uniform(15, 40), 1),
        "humidity":    np.random.randint(30, 90),
        "pressure":    np.random.randint(1005, 1025),
        "wind_speed":  round(np.random.uniform(0.5, 8), 1),
        "wind_deg":    np.random.randint(0, 360),
        "clouds":      np.random.randint(0, 100),
        "weather_main": np.random.choice(["Clear", "Clouds", "Haze", "Mist"]),
    }

def _generate_demo_data(n: int = 500) -> pd.DataFrame:
    """Generate demo training data when real data is not available."""
    np.random.seed(42)
    base = datetime.now() - timedelta(days=30)
    rows = []
    for i in range(n):
        ts = base + timedelta(hours=i)
        w  = _demo_weather()
        # Realistic AQI: higher at peak hours, worse in winter/night
        aqi_base = 120 + 40 * np.sin(ts.hour * np.pi / 12)
        aqi_base += 20 if ts.weekday() < 5 else -10   # worse on weekdays
        aqi_base += np.random.normal(0, 20)
        aqi_base = max(10, min(400, aqi_base))
        feats = compute_features(w, aqi_base, ts)
        rows.append(feats)
    return pd.DataFrame(rows)
