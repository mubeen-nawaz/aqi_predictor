"""
Prediction Module
Predicts the AQI for the next 3 days using the trained model.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.train    import load_model, FEATURE_COLS
from src.features import fetch_weather, _aqi_category, _heat_index, _wind_chill


def predict_next_3_days(city: str = None) -> list[dict]:
    """
    Predict AQI for the next 3 days (72 hours).
    Returns a list of hourly predictions.
    """
    model, scaler, meta = load_model()
    weather = fetch_weather(city) if city else fetch_weather()
    now     = datetime.now()
    predictions = []

    for hour_offset in range(72):  # 3 days = 72 hours
        future_time = now + timedelta(hours=hour_offset)

        # Adjust weather features for the future (simple trend)
        temp_adj = weather["temperature"] + np.sin(future_time.hour * np.pi / 12) * 3
        humid_adj = min(100, weather["humidity"] + hour_offset * 0.1)

        features = {
            "hour":        future_time.hour,
            "day_of_week": future_time.weekday(),
            "month":       future_time.month,
            "is_weekend":  int(future_time.weekday() >= 5),
            "temperature": temp_adj,
            "humidity":    humid_adj,
            "pressure":    weather["pressure"],
            "wind_speed":  weather["wind_speed"],
            "wind_deg":    weather["wind_deg"],
            "clouds":      weather["clouds"],
            "heat_index":  _heat_index(temp_adj, humid_adj),
            "wind_chill":  _wind_chill(temp_adj, weather["wind_speed"]),
            "is_calm":     int(weather["wind_speed"] < 2),
            "is_humid":    int(humid_adj > 80),
        }

        X = np.array([[features[f] for f in FEATURE_COLS]])
        X_scaled = scaler.transform(X)
        aqi_pred = float(model.predict(X_scaled)[0])
        aqi_pred = max(0, round(aqi_pred, 1))

        predictions.append({
            "datetime":    future_time.strftime("%Y-%m-%d %H:%M"),
            "date":        future_time.strftime("%b %d"),
            "hour":        future_time.hour,
            "hour_label":  future_time.strftime("%I %p"),
            "day_label":   future_time.strftime("%A"),
            "aqi":         aqi_pred,
            "category":    _aqi_category(aqi_pred),
            "temperature": round(temp_adj, 1),
            "humidity":    round(humid_adj, 1),
            "health_tip":  _health_tip(aqi_pred),
            "color":       _aqi_color(aqi_pred),
        })

    return predictions


def predict_daily_summary(predictions: list[dict]) -> list[dict]:
    """Group hourly predictions into a daily summary."""
    df = pd.DataFrame(predictions)
    df["date_key"] = pd.to_datetime(df["datetime"]).dt.date
    summary = []
    for date, group in df.groupby("date_key"):
        summary.append({
            "date":      group.iloc[0]["date"],
            "day_label": group.iloc[0]["day_label"],
            "aqi_avg":   round(group["aqi"].mean(), 1),
            "aqi_max":   round(group["aqi"].max(), 1),
            "aqi_min":   round(group["aqi"].min(), 1),
            "category":  _aqi_category(group["aqi"].mean()),
            "color":     _aqi_color(group["aqi"].mean()),
            "health_tip": _health_tip(group["aqi"].mean()),
        })
    return summary[:3]  # Only 3 days


def _health_tip(aqi: float) -> str:
    if aqi <= 50:
        return "✅ Perfectly fine for outdoor activities!"
    if aqi <= 100:
        return "🟡 Sensitive people should take precautions."
    if aqi <= 150:
        return "🟠 Children and the elderly should stay indoors."
    if aqi <= 200:
        return "🔴 Everyone should reduce going outside. Wear a mask."
    if aqi <= 300:
        return "🟣 Stay indoors. Keep windows closed."
    return "⛔ Emergency! Do not go outside at all. Use an air purifier."


def _aqi_color(aqi: float) -> str:
    if aqi <= 50:   return "#00e400"
    if aqi <= 100:  return "#ffff00"
    if aqi <= 150:  return "#ff7e00"
    if aqi <= 200:  return "#ff0000"
    if aqi <= 300:  return "#8f3f97"
    return "#7e0023"


if __name__ == "__main__":
    print("🔮 Next 3 Days AQI Predictions:")
    preds   = predict_next_3_days()
    summary = predict_daily_summary(preds)
    for day in summary:
        print(f"  {day['day_label']:12s}: AQI {day['aqi_avg']:6.1f} ({day['category']})")
        print(f"               {day['health_tip']}")
