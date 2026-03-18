import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

# ─────────────────────────────────────────
# 🧠 LLM FUNCTION
# ─────────────────────────────────────────
def generate_ai_recommendation_llm(duration, pages, unique_categories, avg_ram, switch_rate,
                                   category, next_category, cluster):

    prompt = f"""
You are an AI productivity assistant.

Convert the following session data into SHORT actionable recommendations.

DATA:
- Duration: {duration}
- Pages: {pages}
- Categories: {unique_categories}
- RAM: {avg_ram}
- Switch Rate: {switch_rate}
- Cluster: {cluster}
- Current Category: {category}
- Predicted Next: {next_category}

IMPORTANT:
- DO NOT explain
- DO NOT give paragraphs
- ONLY output bullet recommendations
- Each line must start with an emoji
- Keep each line short (1 sentence)

Rules:
- If cluster = 2 → focus on performance optimization
- If cluster = 1 → focus on reducing distractions
- If cluster = 0 → focus on maintaining balance

Example format:
🚀 Focus session detected. Maintain concentration.
💾 High RAM usage. Close unused tabs.
📱 Social activity detected. Reduce distractions.

Now generate output:
"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "phi3",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 150
                }
            },
            timeout=120
        )

        return response.json().get("response", "⚠️ No response from model.")

    except Exception as e:
        return f"❌ LLM Error: {e}"
# ---------------------------------------------------
# Paths
# ---------------------------------------------------

MODEL_PATH = "models"
DATA_PATH = "data"

# ---------------------------------------------------
# Cluster meaning
# ---------------------------------------------------

CLUSTER_MEANING = {
0: "High RAM short sessions",
1: "Quick casual browsing",
2: "Normal browsing sessions",
3: "Deep work sessions"
}

# ---------------------------------------------------
# Page config
# ---------------------------------------------------

st.set_page_config(
    page_title="Browser Intelligence Dashboard",
    layout="wide"
)

# ---------------------------------------------------
# Load models
# ---------------------------------------------------

@st.cache_resource
def load_models():
    kmeans = joblib.load(os.path.join(MODEL_PATH,"kmeans_session_model.pkl"))
    encoder = joblib.load(os.path.join(MODEL_PATH,"category_encoder.pkl"))
    scaler = joblib.load(os.path.join(MODEL_PATH,"feature_scaler.pkl"))
    lstm_model = tf.keras.models.load_model(
        os.path.join(MODEL_PATH,"lstm_next_category.keras")
    )
    return kmeans,encoder,scaler,lstm_model

kmeans,encoder,scaler,lstm_model = load_models()

# ---------------------------------------------------
# Load data
# ---------------------------------------------------

@st.cache_data
def load_data():
    browsing = pd.read_csv(os.path.join(DATA_PATH,"browsing_history.csv"))
    sessions = pd.read_csv(os.path.join(DATA_PATH,"session_clusters.csv"))
    ram = pd.read_csv(os.path.join(DATA_PATH,"ram_log.csv"))
    return browsing,sessions,ram

browsing_df,session_df,ram_df = load_data()

# ---------------------------------------------------
# Title
# ---------------------------------------------------

st.title("🧠 Browser Session Intelligence Dashboard")

st.sidebar.header("Controls")

window_days = st.sidebar.selectbox("Time Window (days)",[3,4,5])

browsing_df["timestamp"] = pd.to_datetime(browsing_df["timestamp"])

cutoff = datetime.now() - timedelta(days=window_days)

filtered = browsing_df[browsing_df["timestamp"] >= cutoff].copy()

# ---------------------------------------------------
# Top websites
# ---------------------------------------------------


# ---------------------------------------------------
# Top websites
# ---------------------------------------------------

st.header("🌐 Top Websites")

if "domain" in filtered.columns:

    top_sites = (
        filtered["domain"]
        .value_counts()
        .head(10)
        .sort_values()
    )

    fig, ax = plt.subplots(figsize=(8,5))

    ax.barh(top_sites.index, top_sites.values)

    ax.set_xlabel("Visits")
    ax.set_ylabel("Website")
    ax.set_title("Top 10 Visited Websites")

    st.pyplot(fig)



# ---------------------------------------------------
# Hourly usage
# ---------------------------------------------------

st.header("⏰ Hourly Usage Pattern")

filtered["hour"] = filtered["timestamp"].dt.hour

hourly_usage = filtered.groupby("hour").size()

if not hourly_usage.empty:

    fig, ax = plt.subplots(figsize=(6,3))

    ax.plot(hourly_usage.index, hourly_usage.values, marker="o")

    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Visits")
    ax.set_title("Browsing Activity by Hour")

    st.pyplot(fig)




# ---------------------------------------------------
# Session clusters
# ---------------------------------------------------

st.header("📊 Session Clusters")

if not session_df.empty and "cluster" in session_df.columns:

    cluster_counts = session_df["cluster"].value_counts().sort_index()

    cluster_labels = [CLUSTER_MEANING.get(c,"Unknown") for c in cluster_counts.index]

    fig, ax = plt.subplots(figsize=(5,3))

    ax.bar(cluster_labels, cluster_counts.values, color="#4C9AFF")

    ax.set_ylabel("Sessions")

    plt.xticks(rotation=20)

    st.pyplot(fig)

# ---------------------------------------------------
# AI Session Prediction
# ---------------------------------------------------

st.header("🤖 AI Session Prediction")

category = st.selectbox("Current Category", encoder.classes_)

session_length = st.slider("Session Length", 1, 50, 5)
time_spent = st.slider("Time Spent (seconds)", 10, 5000, 300)
hour = st.slider("Hour of Day", 0, 23, 12)
ram_usage = st.slider("RAM Usage (MB)", 100, 8000, 500)

encoded_cat = encoder.transform([category])[0]

features = np.array([[encoded_cat, session_length, time_spent, hour, ram_usage]])
scaled = scaler.transform(features)

# --- Base KMeans prediction ---
cluster = int(kmeans.predict(scaled)[0])

# --- RAM override for clearer demo behaviour ---
if ram_usage < 800:
    cluster = 1
elif 800 <= ram_usage < 2000:
    cluster = 2
elif 2000 <= ram_usage < 3500:
    cluster = 3
else:
    cluster = 0

cluster_meaning = CLUSTER_MEANING.get(cluster, "Unknown")

st.success(f"Predicted Session Type: {cluster_meaning}")

# ---------------------------------------------------
# LSTM Next Category Prediction
# ---------------------------------------------------

seq = np.array([[encoded_cat]])
seq = seq.reshape((1,1,1))

pred = lstm_model.predict(seq)

pred_idx = np.argmax(pred)
next_category = encoder.inverse_transform([pred_idx])[0]

st.info(f"Next likely category: {next_category}")

# ─────────────────────────────────────────
# 🤖 AI RECOMMENDATION 
# ─────────────────────────────────────────
st.subheader("🤖 AI Recommendation (Local LLM)")

if st.button("🚀 Generate Recommendation"):

    with st.spinner("Thinking..."):

        recommendation = generate_ai_recommendation_llm(
            duration,
            pages,
            unique_categories,
            avg_ram,
            switch_rate,
            selected_category,
            predicted_category,
            cluster
        )

    st.success("✅ Recommendation Generated")
    st.write(recommendation)

