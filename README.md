# Browser Analyzer & AI Session Prediction

An AI-powered system that analyzes browser history, detects anomalies, clusters browsing sessions, and predicts the next website category using machine learning and deep learning models.

---

## 🚀 Features

* 📊 **Browser History Analysis**

  * Collect and process browsing data
  * Extract session-based features

* 🤖 **Machine Learning Models**

  * **KMeans Clustering** – identify browsing behavior patterns
  * **Autoencoder** – anomaly detection
  * **LSTM** – predict next browsing category

* 📈 **Interactive Dashboard**

  * Built with Streamlit
  * Visualize browsing sessions and predictions

* 🧠 **AI Session Prediction**

  * Predict the next website category a user might visit
  * Detect unusual browsing behavior

---

## 📁 Project Structure

```
browser_analyzer
│
├── app
│   └── app.py                     # Streamlit dashboard
│
├── data
│   ├── raw
│   │   ├── browsing_history.csv
│   │   └── ram_log.csv
│   │
│   ├── processed
│   │   └── session_clusters.csv
│   │
│   └── mappings
│       └── domain_category_map.csv
│
├── models
│   ├── autoencoder_anomaly.keras
│   ├── lstm_next_category.keras
│   └── kmeans_session_model.pkl
│
├── encoders
│   ├── category_encoder.pkl
│   └── feature_scaler.pkl
│
├── scripts
│   ├── browser_history.py
│   ├── ram_logger.py
│   └── generate_sample.py
│
├── notebooks
│   └── Preprocessing_data.ipynb
│
├── config
│   └── config.yaml
│
└── logs
```

---

## 🧠 Machine Learning Pipeline

1. **Data Collection**

   * Browser history
   * RAM usage logs

2. **Data Preprocessing**

   * Feature extraction
   * Session creation

3. **Clustering**

   * KMeans identifies browsing behavior groups

4. **Anomaly Detection**

   * Autoencoder detects unusual browsing patterns

5. **Next Category Prediction**

   * LSTM predicts the next browsing category

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/browser_analyzer.git
cd browser_analyzer
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

Start the Streamlit dashboard:

```bash
streamlit run app/app.py
```

Then open in your browser:

```
http://localhost:8501
```

---

## 📊 Example Use Cases

* Personal browsing behavior analysis
* Detect suspicious browsing activity
* AI-based web recommendation system
* Research on user browsing patterns

---

## 🛠 Technologies Used

* Python
* Pandas
* Scikit-learn
* TensorFlow / Keras
* Streamlit
* NumPy
* Matplotlib

---

## 📌 Future Improvements

* Real-time browser monitoring
* Deep user behavior modeling
* Reinforcement learning recommendations
* Cloud deployment
* Browser extension integration

---

## 👨‍💻 Author

Vishnu Karthik

---

## 📄 License

This project is licensed under the MIT License.
