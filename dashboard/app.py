"""
AQI Predictor Dashboard
StBeautiful web dashboard with Streamlit - absolutely free!
Run: streamlit run dashboard/app.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json
from datetime import datetime

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pearls AQI Predictor",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

.aqi-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}
.aqi-number { font-size: 72px; font-weight: 700; line-height: 1; }
.aqi-label  { font-size: 14px; color: #888; margin-top: 4px; }
.aqi-cat    { font-size: 20px; font-weight: 600; margin-top: 8px; }

.day-card {
    background: rgba(255,255,255,0.04);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.06);
}
.metric-row { display: flex; gap: 12px; margin-bottom: 12px; }
</style>
""", unsafe_allow_html=True)


# ── Load data helpers ─────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_predictions():
    try:
        from src.predict import predict_next_3_days, predict_daily_summary
        preds   = predict_next_3_days()
        summary = predict_daily_summary(preds)
        return preds, summary, None
    except FileNotFoundError:
        return None, None, "model_not_found"
    except Exception as e:
        return None, None, str(e)


@st.cache_data(ttl=3600)
def load_metadata():
    path = "models/metadata.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


@st.cache_data(ttl=300)
def load_history():
    path = "data/feature_store.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌿 Pearls AQI Predictor")
    st.markdown("---")

    city = st.text_input("🏙️ City", value="Lahore")
    st.markdown("---")

    if st.button("🔄 Update Data", use_container_width=True):
        with st.spinner("Fetching data..."):
            try:
                from src.features import run_feature_pipeline
                run_feature_pipeline()
                st.cache_data.clear()
                st.success("✅ Data updated!")
            except Exception as e:
                st.error(f"Error: {e}")

    if st.button("🤖 Retrain Model", use_container_width=True):
        with st.spinner("Training in progress... (1-2 min)"):
            try:
                from src.train import train
                meta = train()
                st.cache_data.clear()
                st.success(f"✅ Trained! MAE: {meta['metrics']['mae']:.1f}")
            except Exception as e:
                st.error(f"Error: {e}")

    st.markdown("---")
    meta = load_metadata()
    if meta:
        st.markdown("**📊 Model Info**")
        st.markdown(f"- Model: `{meta['model_name']}`")
        st.markdown(f"- MAE: `{meta['metrics']['mae']}`")
        st.markdown(f"- R²: `{meta['metrics']['r2']}`")
        trained = meta['trained_at'][:16].replace('T', ' ')
        st.markdown(f"- Trained: `{trained}`")


# ── Main content ──────────────────────────────────────────────────────────────
st.title("🌿 Pearls AQI Predictor")
st.markdown(f"**{city}** · {datetime.now().strftime('%A, %B %d %Y')}")

preds, summary, error = load_predictions()

if error == "model_not_found":
    st.warning("⚠️ Model not trained yet. Click 'Retrain Model' in the sidebar!")
    st.info("For the first time: click Update Data → Retrain Model in the sidebar")
    st.stop()
elif error:
    st.error(f"Error: {error}")
    st.stop()

# ── Current AQI ───────────────────────────────────────────────────────────────
current = preds[0]
color   = current["color"]

col1, col2, col3, col4 = st.columns([1.5, 1, 1, 1])

with col1:
    st.markdown(f"""
    <div class="aqi-card">
        <div class="aqi-label">CURRENT AQI</div>
        <div class="aqi-number" style="color:{color}">{current['aqi']:.0f}</div>
        <div class="aqi-cat" style="color:{color}">{current['category']}</div>
        <div style="margin-top:12px;font-size:13px;color:#aaa">{current['health_tip']}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.metric("🌡️ Temperature", f"{current['temperature']}°C")
    st.metric("💧 Humidity", f"{current['humidity']}%")

with col3:
    if summary and len(summary) > 0:
        st.metric("📅 Tomorrow AQI", f"{summary[1]['aqi_avg']:.0f}" if len(summary) > 1 else "N/A")
        st.metric("📅 Day After Tomorrow AQI", f"{summary[2]['aqi_avg']:.0f}" if len(summary) > 2 else "N/A")

with col4:
    if meta:
        st.metric("🎯 Model Accuracy", f"R² {meta['metrics']['r2']}")
        st.metric("📉 Avg Error", f"±{meta['metrics']['mae']:.1f} AQI")

st.markdown("---")

# ── 3-Day Forecast Chart ───────────────────────────────────────────────────────
st.subheader("📈 72-Hour Forecast")

df_preds = pd.DataFrame(preds)
df_preds["datetime"] = pd.to_datetime(df_preds["datetime"])

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df_preds["datetime"],
    y=df_preds["aqi"],
    mode="lines",
    line=dict(color="#00d4aa", width=2.5),
    fill="tozeroy",
    fillcolor="rgba(0,212,170,0.1)",
    name="AQI",
))

# AQI level lines
for level, label, clr in [(50,"Good","#00e400"), (100,"Moderate","#ffff00"),
                           (150,"Unhealthy SG","#ff7e00"), (200,"Unhealthy","#ff0000")]:
    fig.add_hline(y=level, line_dash="dot", line_color=clr, opacity=0.5,
                  annotation_text=label, annotation_position="right")

fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    height=350,
    margin=dict(l=0, r=0, t=20, b=0),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="AQI"),
    showlegend=False,
)
st.plotly_chart(fig, use_container_width=True)

# ── Daily Summary Cards ────────────────────────────────────────────────────────
if summary:
    st.subheader("📅 3-Day Summary")
    cols = st.columns(3)
    for i, (col, day) in enumerate(zip(cols, summary)):
        with col:
            st.markdown(f"""
            <div class="day-card">
                <div style="font-weight:700;font-size:16px">{day['day_label']}</div>
                <div style="font-size:12px;color:#888">{day['date']}</div>
                <div style="font-size:48px;font-weight:700;color:{day['color']};margin:8px 0">
                    {day['aqi_avg']:.0f}
                </div>
                <div style="color:{day['color']};font-weight:600">{day['category']}</div>
                <div style="font-size:11px;color:#888;margin-top:8px">
                    Low: {day['aqi_min']:.0f} · High: {day['aqi_max']:.0f}
                </div>
                <div style="font-size:12px;margin-top:8px">{day['health_tip']}</div>
            </div>
            """, unsafe_allow_html=True)

st.markdown("---")

# ── Feature Importance ─────────────────────────────────────────────────────────
if meta and meta.get("feature_importance"):
    st.subheader("🔍 Feature Importance (SHAP-style)")
    fi = meta["feature_importance"]
    fi_df = pd.DataFrame(list(fi.items()), columns=["Feature", "Importance"])
    fi_df = fi_df.sort_values("Importance", ascending=True).tail(10)

    fig2 = px.bar(
        fi_df, x="Importance", y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="teal",
        template="plotly_dark",
    )
    fig2.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=300,
        margin=dict(l=0, r=0, t=20, b=0),
        showlegend=False,
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── Historical Data ────────────────────────────────────────────────────────────
hist = load_history()
if hist is not None and len(hist) > 0:
    st.subheader("📚 Historical AQI Data")
    hist["timestamp"] = pd.to_datetime(hist["timestamp"])
    hist = hist.sort_values("timestamp").tail(200)

    fig3 = px.line(hist, x="timestamp", y="aqi",
                   template="plotly_dark",
                   labels={"aqi": "AQI", "timestamp": "Date"})
    fig3.update_traces(line_color="#7c6af7")
    fig3.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=250,
        margin=dict(l=0, r=0, t=20, b=0),
    )
    st.plotly_chart(fig3, use_container_width=True)

# ── AQI Reference Guide ────────────────────────────────────────────────────────
with st.expander("📖 AQI Reference Guide"):
    guide = pd.DataFrame([
        {"Range": "0-50",   "Category": "Good",                    "Color": "🟢", "Health": "Safe for outdoor activities"},
        {"Range": "51-100", "Category": "Moderate",                "Color": "🟡", "Health": "Caution for sensitive groups"},
        {"Range": "101-150","Category": "Unhealthy for Sensitive",  "Color": "🟠", "Health": "Caution for children and elderly"},
        {"Range": "151-200","Category": "Unhealthy",               "Color": "🔴", "Health": "Caution for everyone, wear a mask"},
        {"Range": "201-300","Category": "Very Unhealthy",          "Color": "🟣", "Health": "Stay indoors"},
        {"Range": "301+",   "Category": "Hazardous",               "Color": "⛔", "Health": "Emergency - do not go outside"},
