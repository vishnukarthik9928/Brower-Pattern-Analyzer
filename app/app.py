import streamlit as st
import numpy as np
import requests
import pandas as pd
import matplotlib.pyplot as plt

# 🔥 IMPORT GENERATORS
from generate_sample import (
    generate_browsing_history,
    generate_ram_log,
    generate_domain_category_map
)

# 🔥 IMPORT PIPELINE FUNCTIONS
from main_pipeline import (
    load_data,
    merge_data,
    create_sessions,
    session_features,
    clustering,
    anomaly_detection,
    sequence_model,
    save_all
)

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
- ONLY bullet points
- Each line starts with emoji
- Keep short

Rules:
- Cluster 2 → performance
- Cluster 1 → reduce distractions
- Cluster 0 → maintain balance
"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "phi3",
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )

        return response.json().get("response", "⚠️ No response")

    except Exception as e:
        return f"❌ LLM Error: {e}"


# ─────────────────────────────────────────
# 📋 CONFIG
# ─────────────────────────────────────────
st.set_page_config(page_title="AI Dashboard", layout="wide")
st.title("🧠 AI-Powered Session Analysis Dashboard")

# ─────────────────────────────────────────
# 🔘 GENERATE + LOAD SECTION
# ─────────────────────────────────────────
st.subheader("⚙️ Data Controls")

col1, col2 = st.columns(2)

# ✅ GENERATE BUTTON
if col1.button("🔄 Generate Sample Data"):
    with st.spinner("Generating data..."):
        generate_browsing_history(days=5)
        generate_ram_log(days=5)
        generate_domain_category_map()

    st.success("✅ Sample data generated successfully!")

# ✅ LOAD + FULL PIPELINE BUTTON
if col2.button("📂 Load & Process Data"):

    try:
        with st.spinner("Running full pipeline..."):

            # 🔥 STEP 1: LOAD
            history, ram = load_data()

            # 🔥 STEP 2: MERGE
            merged = merge_data(history, ram)

            # 🔥 STEP 3: SESSIONS
            merged = create_sessions(merged)

            # 🔥 STEP 4: FEATURES
            sessions = session_features(merged)

            # 🔥 STEP 5: CLUSTERING
            sessions, scaler, kmeans = clustering(sessions)

            # 🔥 STEP 6: ANOMALY
            features = sessions[["duration", "pages", "unique_domains", "avg_ram"]]
            autoencoder, anomalies = anomaly_detection(features)
            sessions["anomaly"] = anomalies

            # 🔥 STEP 7: LSTM
            lstm_model, encoder = sequence_model(merged)

            # 🔥 STEP 8: SAVE
            save_all(sessions, scaler, kmeans, autoencoder, lstm_model, encoder)

            # Save merged also for visualization
            merged.to_csv("merged_data.csv", index=False)

        st.success("✅ Data processed successfully!")

    except Exception as e:
        st.error(f"❌ Pipeline Error: {e}")


# ─────────────────────────────────────────
# 📊 INPUTS
# ─────────────────────────────────────────
st.sidebar.header("📊 Session Features")

duration = st.sidebar.slider("Duration", 0.0, 300.0, 30.0)
pages = st.sidebar.slider("Pages", 1, 100, 10)
unique_categories = st.sidebar.slider("Categories", 1, 20, 5)
avg_ram = st.sidebar.slider("RAM (MB)", 100.0, 4000.0, 800.0)
switch_rate = st.sidebar.slider("Switch Rate", 0.0, 1.0, 0.5)

# ─────────────────────────────────────────
# 🔮 NEXT CATEGORY
# ─────────────────────────────────────────
CATEGORY_FLOW = {
    'coding': 'learning',
    'learning': 'coding',
    'social': 'entertainment',
    'entertainment': 'social',
    'work': 'utility',
    'utility': 'work'
}

selected_category = st.selectbox("Category", list(CATEGORY_FLOW.keys()))
predicted_category = CATEGORY_FLOW[selected_category]

st.success(f"Next: {predicted_category}")

# ─────────────────────────────────────────
# 🚨 ANOMALY
# ─────────────────────────────────────────
error = (avg_ram / 3000) + (switch_rate * 0.8)

if error > 1.2:
    st.error("⚠️ Anomaly")
else:
    st.success("✅ Normal")

# ─────────────────────────────────────────
# 📌 CLUSTER
# ─────────────────────────────────────────
if avg_ram > 2500:
    cluster = 2
elif switch_rate > 0.7:
    cluster = 1
else:
    cluster = 0

cluster_names = {
    0: "Low Usage",
    1: "High Switching",
    2: "High RAM Usage"
}

cluster_label = cluster_names.get(cluster, "Unknown")
st.metric("Cluster", cluster_label)

# ─────────────────────────────────────────
# 📊 DATA VISUALIZATION
# ─────────────────────────────────────────
st.subheader("📊 Insights")

FIG_SIZE = (5, 3)

try:
    df = pd.read_csv("merged_data.csv")

    if 'domain' in df.columns:
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        df['domain'].value_counts().head(10).plot(kind='bar', ax=ax)
        ax.set_title("Top Domains")
        st.pyplot(fig)

    if 'category' in df.columns:
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        df['category'].value_counts().head(10).plot(kind='bar', ax=ax)
        ax.set_title("Categories")
        st.pyplot(fig)

    if 'hour' in df.columns:
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        df['hour'].value_counts().sort_index().plot(ax=ax)
        ax.set_title("Hourly Activity")
        st.pyplot(fig)

    if 'ram_usage' in df.columns:
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        df['ram_usage'].plot(ax=ax)
        ax.set_title("RAM Usage Over Time")
        st.pyplot(fig)

    if 'cpu_percent' in df.columns:
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        df['cpu_percent'].plot(ax=ax)
        ax.set_title("CPU Usage Over Time")
        st.pyplot(fig)

except Exception as e:
    st.warning(f"⚠️ Error: {e}")

# ─────────────────────────────────────────
# 🤖 AI RECOMMENDATION
# ─────────────────────────────────────────
if st.button("Generate AI Recommendation"):

    with st.spinner("Thinking..."):
        result = generate_ai_recommendation_llm(
            duration, pages, unique_categories,
            avg_ram, switch_rate,
            selected_category, predicted_category, cluster
        )

    st.write(result)
