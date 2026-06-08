"""Basic tests for AQI Predictor pipeline.

Translated to English.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import pandas as pd
from src.features import compute_features, _aqi_category, _generate_demo_data


def test_aqi_categories():
    assert _aqi_category(25)  == "Good"
    assert _aqi_category(75)  == "Moderate"
    assert _aqi_category(125) == "Unhealthy for Sensitive"
    assert _aqi_category(175) == "Unhealthy"
    assert _aqi_category(250) == "Very Unhealthy"
    assert _aqi_category(350) == "Hazardous"


def test_compute_features():
    weather = {
        "temperature": 30, "humidity": 65, "pressure": 1013,
        "wind_speed": 3, "wind_deg": 180, "clouds": 20, "weather_main": "Clear"
    }
    feats = compute_features(weather, aqi=120.0)
    assert "aqi" in feats
    assert feats["aqi"] == 120.0
    assert feats["is_weekend"] in (0, 1)
    assert "heat_index" in feats


def test_demo_data_generation():
    df = _generate_demo_data(n=100)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 100
    assert "aqi" in df.columns
    assert "temperature" in df.columns


def test_training_pipeline():
    from src.train import train
    meta = train()
    assert "metrics" in meta
    assert meta["metrics"]["r2"] > 0
    assert os.path.exists("models/best_model.joblib")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
