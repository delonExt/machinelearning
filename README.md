<![CDATA[<div align="center">

# 🧠 YeoCycles — Machine Learning Service

### Menstrual Health Companion · LSTM Deep Learning Prediction

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-3.1-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Microservice prediksi siklus menstruasi** menggunakan model **LSTM (Long Short-Term Memory)** deep learning, dibangun dengan **Flask** dan **TensorFlow** — didukung oleh pipeline data science yang komprehensif.

[Arsitektur](#️-arsitektur-keseluruhan) · [AI Engineering](README_AI_ENGINEERING.md) · [Data Science](README_DATA_SCIENCE.md) · [Quick Start](#-quick-start) · [API Reference](#-api-reference)

</div>

---

## 📚 Dokumentasi Repository

Repository ini memiliki **3 README** yang terpisah untuk kejelasan:

| Dokumen | Deskripsi | Link |
|---------|-----------|------|
| **README.md** | Overview umum, quick start, API reference (file ini) | Anda di sini |
| **README_AI_ENGINEERING.md** | Arsitektur model LSTM, custom components, training, deployment | [→ AI Engineering](README_AI_ENGINEERING.md) |
| **README_DATA_SCIENCE.md** | EDA, preprocessing, analisis statistik, Streamlit dashboard | [→ Data Science](README_DATA_SCIENCE.md) |

---

## 🏗️ Arsitektur Keseluruhan

### Full-Stack System

<p align="center">
  <img src="docs/images/system_architecture.png" alt="System Architecture" width="700" />
</p>

### ML Service dalam Konteks Sistem

```mermaid
graph TB
    subgraph Frontend["🖥️ Frontend (React + Vite)"]
        Dashboard["📊 Dashboard"]
        Predictions["🔮 Prediction UI"]
    end

    subgraph Backend["⚙️ Backend (Express.js)"]
        PredRoute["predictions.js<br/>GET /api/predictions"]
        DB[("🗄️ MySQL<br/>menstrual_cycles<br/>predictions")]
    end

    subgraph MLService["🧠 ML Service (Flask + TensorFlow)"]
        direction TB
        FlaskAPI["Flask API Server<br/>Port 5001"]
        
        subgraph ModelPipeline["Model Pipeline"]
            Preprocess["🔄 Preprocessing<br/>MinMaxScaler, Padding"]
            LSTM["🧠 LSTM Model<br/>64→32 units + Attention"]
            PostProcess["📐 Post-processing<br/>Inverse scale, clamp 21-45"]
        end
        
        subgraph Artifacts["Model Artifacts"]
            H5["lstm_model.h5"]
            ScalerX["scaler_X.pkl"]
            ScalerY["scaler_y.pkl"]
        end
    end

    Dashboard --> PredRoute
    PredRoute -->|"Fetch last 3 cycles"| DB
    PredRoute -->|"POST /predict"| FlaskAPI
    FlaskAPI --> Preprocess
    Preprocess --> LSTM
    LSTM --> PostProcess
    PostProcess -->|"{ predicted_cycle_length, confidence }"| PredRoute
    PredRoute -->|"Save prediction"| DB
    PredRoute -->|"Return to client"| Dashboard
    
    LSTM -.->|"Load"| Artifacts

    style Frontend fill:#1a1a2e,stroke:#ec4899,color:#fff
    style Backend fill:#1a1a2e,stroke:#a855f7,color:#fff
    style MLService fill:#1a1a2e,stroke:#10b981,color:#fff
    style ModelPipeline fill:#0f2027,stroke:#3b82f6,color:#fff
```

### Klasifikasi: Machine Learning vs Deep Learning

> **❓ Apakah ini Machine Learning atau Deep Learning?**

**Jawaban: Ini adalah Deep Learning** — spesifik nya menggunakan arsitektur **LSTM (Long Short-Term Memory)** yang merupakan bagian dari keluarga **Recurrent Neural Networks (RNN)**.

```mermaid
graph TD
    AI["🤖 Artificial Intelligence"] --> ML["📊 Machine Learning"]
    ML --> Traditional["📈 Traditional ML<br/>Linear Regression, SVM,<br/>Decision Trees, Random Forest"]
    ML --> DL["🧠 Deep Learning"]
    DL --> CNN["🖼️ CNN<br/>Convolutional Neural Networks"]
    DL --> RNN["🔄 RNN<br/>Recurrent Neural Networks"]
    DL --> Transformer["📝 Transformers<br/>BERT, GPT"]
    RNN --> LSTM["⭐ LSTM<br/>Long Short-Term Memory<br/>(YANG KITA GUNAKAN)"]
    RNN --> GRU["GRU<br/>Gated Recurrent Units"]
    
    style LSTM fill:#ec4899,stroke:#fff,color:#fff,stroke-width:3px
    style DL fill:#a855f7,stroke:#fff,color:#fff
    style RNN fill:#6366f1,stroke:#fff,color:#fff
```

| Aspek | Machine Learning (Traditional) | Deep Learning (Kita) |
|-------|-------------------------------|---------------------|
| **Model** | Linear Regression, SVM, Random Forest | LSTM Neural Network |
| **Feature Engineering** | Manual, dominan | Otomatis + manual hybrid |
| **Data Requirement** | Kecil–sedang | Sedang–besar |
| **Architecture** | Shallow (1–2 layers) | Deep (multi-layer, 64→32→16→1) |
| **Sequential Data** | Tidak native | ✅ Native (timestep-aware) |
| **Framework** | scikit-learn | TensorFlow/Keras |

---

## 🛠️ Tech Stack

| Technology | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.10+ | Runtime |
| **Flask** | 3.1.0 | REST API framework |
| **Flask-CORS** | 5.0.0 | Cross-origin support |
| **TensorFlow** | 2.16.1 | Deep learning framework |
| **Keras** | (built-in) | High-level neural network API |
| **scikit-learn** | 1.4.0 | Preprocessing (MinMaxScaler) |
| **Pandas** | 2.2.0 | Data manipulation |
| **NumPy** | 1.26.4 | Numerical computing |
| **joblib** | 1.3.2 | Serialization (scalers) |
| **Matplotlib** | 3.8.x | Static visualizations (EDA) |
| **Seaborn** | 0.13.x | Statistical visualizations |
| **Plotly** | 5.x | Interactive visualizations |
| **Streamlit** | 1.31.x | Data dashboard |
| **SciPy** | 1.12.x | Statistical tests |

---

## 🚀 Quick Start

```bash
# 1. Clone repository
git clone https://github.com/Coding-Camp-Capstone-Project-2026/machinelearning.git
cd machinelearning

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Opsional) Re-train model
python train.py

# 4. Start prediction API
python app.py
# → http://localhost:5001

# 5. (Opsional) Jalankan EDA analysis
python eda_analysis.py

# 6. (Opsional) Jalankan Streamlit dashboard
streamlit run streamlit_app.py
```

---

## 📁 Project Structure

```
machinelearning/
├── README.md                    # 📖 Overview (file ini)
├── README_AI_ENGINEERING.md     # 🧠 AI/Deep Learning documentation
├── README_DATA_SCIENCE.md       # 📊 Data Science documentation
├── requirements.txt             # 📦 Python dependencies
│
├── app.py                       # 🚀 Flask API server (prediction endpoint)
├── train.py                     # 🏋️ LSTM model training script
├── preprocess.py                # 🔄 Data preprocessing pipeline
├── custom_components.py         # ⚙️ Custom Keras layers, losses, callbacks
│
├── eda_analysis.py              # 📊 Exploratory Data Analysis script
├── streamlit_app.py             # 📈 Interactive Streamlit dashboard
│
├── data_dictionary.md           # 📖 Dataset column documentation
├── problem_discovery.md         # 🔍 Problem statement & methodology
│
├── data/
│   └── sample_data.csv          # 📊 Training dataset (95 records, 8 users)
│
├── model/
│   ├── lstm_model.h5            # 🧠 Trained model (HDF5) — gitignored
│   ├── lstm_model.keras         # 🧠 Trained model (Keras format)
│   ├── scaler_X.pkl             # 📐 Feature scaler
│   └── scaler_y.pkl             # 📐 Target scaler
│
├── analysis_output/             # 📊 EDA visualization outputs (11 PNG files)
│   ├── 01_distribusi_cycle_length.png
│   ├── 02_boxplot_per_user.png
│   ├── 03_correlation_heatmap.png
│   ├── ...
│   └── 11_bq5_distribusi_keseluruhan.png
│
├── logs/                        # 📝 Training logs
│   └── training_log.csv         # Epoch-by-epoch metrics
│
└── docs/
    └── images/                  # 📷 README visuals
```

---

## 🔌 API Reference

### Health Check

```http
GET /health
```

**Response:**
```json
{ "status": "ok", "model_loaded": true }
```

### Prediction

```http
POST /predict
Content-Type: application/json
```

**Request:**
```json
{
  "cycles": [28, 30, 27],
  "sleep": [4, 3, 4],
  "stress": [2, 3, 2],
  "fasting": [0, 5, 0]
}
```

**Response (Model Loaded):**
```json
{
  "predicted_cycle_length": 28,
  "confidence": 0.85,
  "model_version": "1.0"
}
```

**Response (Fallback):**
```json
{
  "predicted_cycle_length": 28,
  "confidence": 0.5,
  "model_version": "fallback_average",
  "message": "Model not loaded. Using simple average."
}
```

### Prediction Flow

```mermaid
sequenceDiagram
    participant C as ⚙️ Backend
    participant F as 🚀 Flask API
    participant P as 🔄 Preprocessor
    participant M as 🧠 LSTM Model
    participant S as 📐 Scalers

    C->>F: POST /predict { cycles, sleep, stress, fasting }
    F->>F: Validate input (cycles required)
    
    alt Model Loaded
        F->>P: preprocess_for_prediction(data)
        P->>P: Pad to 3 timesteps if needed
        P->>P: Estimate period_length
        P->>S: Load scaler_X.pkl → transform features
        P->>P: Reshape to (1, 3, 5) 3D tensor
        P-->>F: preprocessed_input
        
        F->>M: model.predict(input)
        M-->>F: raw_prediction (scaled 0-1)
        
        F->>S: Load scaler_y.pkl → inverse_transform
        F->>F: Clamp result to 21-45 days
        F->>F: Calculate confidence score
        F-->>C: { predicted_cycle_length, confidence, model_version: "1.0" }
    else Model Not Loaded
        F->>F: Calculate simple average of cycles[]
        F-->>C: { prediction, confidence: 0.5, model_version: "fallback" }
    end
```

### Confidence Calculation

```
base_confidence = 1.0 - (std_deviation(cycles) / 10.0)
data_factor = min(1.0, num_cycles / 6.0)
final_confidence = clamp(base_confidence × data_factor, 0.3, 0.95)
```

### Fallback Mechanism

```mermaid
graph TD
    A["POST /predict"] --> B{"Model loaded?"}
    B -->|"Yes"| C["LSTM Prediction"]
    B -->|"No"| D{"Cycles data?"}
    
    C --> E{"Prediction error?"}
    E -->|"No"| F["✅ Return prediction<br/>confidence: 0.3-0.95"]
    E -->|"Yes"| G["⚠️ Return 28 days<br/>confidence: 0.3"]
    
    D -->|"Yes"| H["📊 Simple average<br/>confidence: 0.5"]
    D -->|"No"| I["❌ Return 28 days<br/>confidence: 0.3"]

    style F fill:#10b981,stroke:#fff,color:#fff
    style H fill:#f59e0b,stroke:#fff,color:#fff
    style G fill:#ef4444,stroke:#fff,color:#fff
    style I fill:#ef4444,stroke:#fff,color:#fff
```

---

## 📊 Model Files

| File | Size | Description |
|------|------|-------------|
| `lstm_model.h5` | ~414 KB | Trained LSTM model (HDF5 format) |
| `lstm_model.keras` | ~420 KB | Trained model (Keras native format) |
| `scaler_X.pkl` | ~1 KB | Feature scaler (MinMaxScaler) |
| `scaler_y.pkl` | ~1 KB | Target scaler (MinMaxScaler) |

> **Note**: `.h5` model di-gitignore (besar). `.keras` format disertakan.

---

## 🔗 Related Repositories

| Repository | Description | Link |
|------------|-------------|------|
| **Frontend** | React SPA + Premium UI | [frontend](https://github.com/Coding-Camp-Capstone-Project-2026/frontend) |
| **Backend** | Express.js REST API | [backend](https://github.com/Coding-Camp-Capstone-Project-2026/backend) |
| **Machine Learning** | Flask + LSTM (this repo) | [machinelearning](https://github.com/Coding-Camp-Capstone-Project-2026/machinelearning) |

---

## 📖 Dokumentasi Lengkap

Untuk detail mendalam, lihat:

- 🧠 **[README_AI_ENGINEERING.md](README_AI_ENGINEERING.md)** — Arsitektur LSTM, custom Attention Layer, custom Huber Loss, training pipeline, hyperparameters
- 📊 **[README_DATA_SCIENCE.md](README_DATA_SCIENCE.md)** — EDA pipeline, business questions, statistical analysis, Streamlit dashboard, visualisasi

---

## 👥 Tim Pengembang

Dibuat oleh **Ridho dan teman-teman** — Capstone Project Coding Camp 2026

**Powered by [kamidukung.biz.id](https://kamidukung.biz.id/)**
]]>
