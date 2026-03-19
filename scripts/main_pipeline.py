import os
import pandas as pd
import numpy as np
from datetime import timedelta

# ML
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans

# Deep Learning
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Bidirectional

import joblib


# =========================
# 📁 PATH SETUP
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

HISTORY_PATH = os.path.join(DATA_DIR, "browsing_history.csv")
RAM_PATH = os.path.join(DATA_DIR, "ram_log.csv")


# =========================
# 1. LOAD DATA
# =========================
def load_data():
    if not os.path.exists(HISTORY_PATH) or not os.path.exists(RAM_PATH):
        raise FileNotFoundError("❌ Data files not found in src/data/")

    history = pd.read_csv(HISTORY_PATH)
    ram = pd.read_csv(RAM_PATH)

    history["timestamp"] = pd.to_datetime(history["timestamp"])
    ram["timestamp"] = pd.to_datetime(ram["timestamp"])

    history = history.sort_values("timestamp")
    ram = ram.sort_values("timestamp")

    return history, ram


# =========================
# 2. MERGE DATA
# =========================
def merge_data(history, ram):
    merged = pd.merge_asof(
        history,
        ram,
        on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta("1min")
    )

    # ✅ Fix missing values
    merged["ram_usage"].fillna(method="ffill", inplace=True)
    merged["cpu_percent"].fillna(method="ffill", inplace=True)

    merged["time_diff"] = merged["timestamp"].diff().dt.total_seconds().fillna(0)

    return merged


# =========================
# 3. SESSION CREATION
# =========================
def create_sessions(df):
    SESSION_GAP = 15 * 60

    df["new_session"] = (df["time_diff"] > SESSION_GAP).astype(int)
    df["session_id"] = df["new_session"].cumsum()

    return df


# =========================
# 4. FEATURE ENGINEERING
# =========================
def session_features(df):
    sessions = df.groupby("session_id").agg({
        "timestamp": ["min", "max", "count"],
        "domain": "nunique",
        "ram_usage": "mean",
        "cpu_percent": "mean",
        "category": lambda x: x.mode()[0] if len(x.mode()) > 0 else "unknown"
    })

    sessions.columns = [
        "start_time", "end_time", "pages",
        "unique_domains", "avg_ram", "avg_cpu", "top_category"
    ]

    sessions = sessions.reset_index()

    sessions["duration"] = (
        sessions["end_time"] - sessions["start_time"]
    ).dt.total_seconds() / 60

    # ✅ Remove invalid sessions
    sessions = sessions[sessions["duration"] > 0]

    return sessions


# =========================
# 5. CLUSTERING
# =========================
def clustering(sessions):
    features = sessions[["duration", "pages", "unique_domains", "avg_ram"]]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=4, random_state=42)
    sessions["cluster"] = kmeans.fit_predict(X_scaled)

    return sessions, scaler, kmeans


# =========================
# 6. AUTOENCODER
# =========================
def anomaly_detection(features):
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    input_dim = X.shape[1]

    input_layer = Input(shape=(input_dim,))
    encoded = Dense(8, activation="relu")(input_layer)
    encoded = Dense(4, activation="relu")(encoded)
    decoded = Dense(8, activation="relu")(encoded)
    decoded = Dense(input_dim, activation="linear")(decoded)

    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer="adam", loss="mse")

    autoencoder.fit(X, X, epochs=10, batch_size=16, verbose=0)

    recon = autoencoder.predict(X)
    mse = np.mean(np.power(X - recon, 2), axis=1)

    threshold = np.percentile(mse, 95)
    anomalies = mse > threshold

    return autoencoder, anomalies


# =========================
# 7. LSTM MODEL
# =========================
def sequence_model(df):
    le = LabelEncoder()
    df["category_id"] = le.fit_transform(df["category"])

    SEQ_LEN = 5

    X, y = [], []
    categories = df["category_id"].values

    for i in range(len(categories) - SEQ_LEN):
        X.append(categories[i:i+SEQ_LEN])
        y.append(categories[i+SEQ_LEN])

    X = np.array(X)
    y = np.array(y)

    # ✅ Safety check
    if len(X) == 0:
        print("⚠️ Not enough data for LSTM")
        return None, le

    model = Sequential([
        Embedding(input_dim=len(le.classes_), output_dim=32),
        Bidirectional(LSTM(64, return_sequences=True)),
        LSTM(32),
        Dense(len(le.classes_), activation="softmax")
    ])

    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)

    return model, le


# =========================
# 8. SAVE OUTPUTS
# =========================
def save_all(sessions, scaler, kmeans, autoencoder, lstm_model, encoder):

    sessions.to_csv(os.path.join(DATA_DIR, "session_output.csv"), index=False)

    joblib.dump(kmeans, os.path.join(DATA_DIR, "kmeans.pkl"))
    joblib.dump(scaler, os.path.join(DATA_DIR, "scaler.pkl"))
    joblib.dump(encoder, os.path.join(DATA_DIR, "label_encoder.pkl"))

    autoencoder.save(os.path.join(DATA_DIR, "autoencoder.h5"))

    if lstm_model:
        lstm_model.save(os.path.join(DATA_DIR, "lstm_model.h5"))

    print("✅ All outputs saved in src/data/")


# =========================
# MAIN
# =========================
def main():
    print("🚀 Starting pipeline...")

    history, ram = load_data()
    merged = merge_data(history, ram)
    merged = create_sessions(merged)

    sessions = session_features(merged)

    sessions, scaler, kmeans = clustering(sessions)

    features = sessions[["duration", "pages", "unique_domains", "avg_ram"]]
    autoencoder, anomalies = anomaly_detection(features)

    sessions["anomaly"] = anomalies

    lstm_model, encoder = sequence_model(merged)

    save_all(sessions, scaler, kmeans, autoencoder, lstm_model, encoder)

    print("🎯 Done!")


if __name__ == "__main__":
    main()
