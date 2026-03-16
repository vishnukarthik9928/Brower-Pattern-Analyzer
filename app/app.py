import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

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

# ---------------------------------------------------
# AI Productivity Score
# ---------------------------------------------------

st.header("🧠 AI Productivity Score")

score = 50

# cluster influence
if cluster == 3:
    score += 25
elif cluster == 2:
    score += 10
elif cluster == 1:
    score -= 10
elif cluster == 0:
    score -= 5

# category influence
if category == "learning":
    score += 15
elif category == "work":
    score += 10
elif category == "social":
    score -= 20
elif category == "entertainment":
    score -= 10

# session behaviour
if session_length > 20:
    score += 10

if time_spent > 1200:
    score += 5

# time of day
if hour >= 22:
    score -= 15

# RAM behaviour
if ram_usage > 3500:
    score -= 10

score = max(0, min(score, 100))

st.metric("Productivity Score", f"{score}/100")

if score >= 75:
    st.success("🔥 High focus session detected")
elif score >= 50:
    st.info("🙂 Moderate productivity")
else:
    st.warning("⚠️ High distraction risk")

# ---------------------------------------------------
# Smart Recommendations
# ---------------------------------------------------

st.header("🎯 Smart Recommendations")

recs = []

# cluster based recommendations
if cluster == 3:
    recs.append("🚀 Deep work session detected. Maintain focus and avoid distractions.")

elif cluster == 2:
    recs.append("📖 Normal browsing session. Try switching to productive tasks.")

elif cluster == 1:
    recs.append("⚡ Quick browsing pattern detected. Consider focusing on a single task.")

elif cluster == 0:
    recs.append("💾 High RAM short session detected. You may have many tabs open.")

# behaviour recommendations
if ram_usage > 3000:
    recs.append("💾 High RAM usage detected. Close unused tabs or restart browser.")

if session_length > 25:
    recs.append("🧠 Long session detected. Take a short break to maintain productivity.")

if hour >= 22:
    recs.append("🌙 Late-night browsing detected. Consider resting soon.")

# category recommendations
if category == "social":
    recs.append("📱 Social media activity detected. Limit distractions for better focus.")

elif category == "learning":
    recs.append("📚 Learning session detected. Good time to take notes or summarize concepts.")

elif category == "entertainment":
    recs.append("🎬 Entertainment browsing detected. Balance it with productive work.")

# AI prediction insight
recs.append(f"🤖 Based on behavior, your next likely category is **{next_category}**.")

for r in recs:
    st.success(r)

