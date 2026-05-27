<![CDATA[<div align="center">

# 🧠 AI Engineering — YeoCycles ML Service

### Deep Learning Model Architecture, Training & Deployment

[← Kembali ke Overview](README.md) · [Data Science →](README_DATA_SCIENCE.md)

</div>

---

## 📑 Daftar Isi

- [Ringkasan Model](#-ringkasan-model)
- [Arsitektur LSTM](#-arsitektur-lstm)
- [Custom Components](#️-custom-components)
- [Training Pipeline](#-training-pipeline)
- [Preprocessing](#-preprocessing-pipeline)
- [Inference & Deployment](#-inference--deployment)
- [Model Evaluation](#-model-evaluation)
- [Hyperparameter Tuning](#-hyperparameter-tuning)

---

## 📋 Ringkasan Model

| Aspek | Detail |
|-------|--------|
| **Tipe Model** | Deep Learning — LSTM (Long Short-Term Memory) |
| **Framework** | TensorFlow 2.16 + Keras (Functional API) |
| **Arsitektur** | 2-layer stacked LSTM + Custom Attention + Dense |
| **Input** | Sequence 3 timesteps × 5 features |
| **Output** | 1 value — predicted cycle length (21–45 hari) |
| **Loss Function** | Custom Weighted Huber Loss |
| **Optimizer** | Adam (lr=0.001 → adaptive reduction) |
| **Total Parameters** | ~25,000 trainable parameters |
| **Training Data** | 95 records, 8 users, synthetic medical-pattern data |
| **Custom Components** | 3 — AttentionLayer, CyclePredictionLoss, TrainingMonitorCallback |

---

## 🏛️ Arsitektur LSTM

### Visual Architecture

<p align="center">
  <img src="docs/images/lstm_architecture.png" alt="LSTM Architecture" width="600" />
</p>

### Layer-by-Layer Breakdown

```mermaid
graph TB
    subgraph InputSection["📥 Input Section"]
        Input["Input Layer<br/>Shape: (3, 5)<br/>3 timesteps × 5 features"]
        Features["Features per timestep:<br/>cycle_length | period_length<br/>avg_sleep | avg_stress | fasting_days"]
    end
    
    subgraph LSTMStack["🧠 LSTM Stack"]
        LSTM1["LSTM Layer 1<br/>64 units<br/>return_sequences=True<br/>Outputs: (batch, 3, 64)"]
        Drop1["Dropout 0.2<br/>Regularization"]
        LSTM2["LSTM Layer 2<br/>32 units<br/>return_sequences=True<br/>Outputs: (batch, 3, 32)"]
        Drop2["Dropout 0.2<br/>Regularization"]
    end
    
    subgraph AttentionSection["⭐ Custom Attention"]
        Attn["AttentionLayer<br/>32 units<br/>Bahdanau-style additive attention<br/>Outputs: (batch, 32)"]
    end
    
    subgraph OutputSection["📤 Output Section"]
        Dense["Dense Layer<br/>16 units, ReLU activation"]
        Output["Output Layer<br/>1 unit, linear activation<br/>predicted cycle length"]
    end
    
    Input --> LSTM1
    Features -.-> Input
    LSTM1 --> Drop1
    Drop1 --> LSTM2
    LSTM2 --> Drop2
    Drop2 --> Attn
    Attn --> Dense
    Dense --> Output
    
    style InputSection fill:#1a1a2e,stroke:#6366f1,color:#fff
    style LSTMStack fill:#1a1a2e,stroke:#ec4899,color:#fff
    style AttentionSection fill:#1a1a2e,stroke:#f59e0b,color:#fff
    style OutputSection fill:#1a1a2e,stroke:#10b981,color:#fff
```

### Mengapa LSTM?

LSTM dipilih karena menstrual cycle data bersifat **sekuensial dan temporal**:

```mermaid
graph LR
    subgraph TimeSequence["📅 Sequence Data (3 Siklus Terakhir)"]
        T1["Siklus 1<br/>28 hari<br/>sleep: 7<br/>stress: 2"]
        T2["Siklus 2<br/>30 hari<br/>sleep: 5<br/>stress: 4"]
        T3["Siklus 3<br/>27 hari<br/>sleep: 6<br/>stress: 3"]
    end
    
    T1 -->|"timestep 1"| LSTM["🧠 LSTM<br/>Ingat pola<br/>jangka panjang"]
    T2 -->|"timestep 2"| LSTM
    T3 -->|"timestep 3"| LSTM
    LSTM -->|"Prediksi"| Pred["🔮 Siklus 4<br/>≈ 28 hari<br/>confidence: 85%"]
    
    style TimeSequence fill:#0f0f23,stroke:#a855f7,color:#fff
    style LSTM fill:#ec4899,stroke:#fff,color:#fff
    style Pred fill:#10b981,stroke:#fff,color:#fff
```

| Keunggulan LSTM | Penjelasan |
|-----------------|-----------|
| **Memory Cell** | Bisa mengingat pola siklus dari siklus-siklus sebelumnya |
| **Forget Gate** | Bisa "melupakan" siklus yang anomali/outlier |
| **Input Gate** | Menentukan informasi baru mana yang penting |
| **Output Gate** | Mengontrol informasi yang diteruskan ke prediksi |
| **Sequence-Aware** | Native support untuk data berurutan (time-series) |

### LSTM Cell — Internal Mechanism

```mermaid
graph LR
    subgraph LSTMCell["LSTM Cell at timestep t"]
        direction TB
        
        xt["x(t)<br/>Input features<br/>(5 values)"]
        ht_prev["h(t-1)<br/>Previous hidden<br/>state"]
        ct_prev["c(t-1)<br/>Previous cell<br/>state"]
        
        fg["🚪 Forget Gate<br/>σ(W_f · [h(t-1), x(t)] + b_f)<br/>Apa yang dilupakan?"]
        ig["🚪 Input Gate<br/>σ(W_i · [h(t-1), x(t)] + b_i)<br/>Apa yang disimpan?"]
        cand["📝 Candidate<br/>tanh(W_c · [h(t-1), x(t)] + b_c)<br/>Informasi baru"]
        og["🚪 Output Gate<br/>σ(W_o · [h(t-1), x(t)] + b_o)<br/>Apa yang di-output?"]
        
        ct["c(t)<br/>New cell state"]
        ht["h(t)<br/>New hidden state<br/>= output"]
        
        xt --> fg
        xt --> ig
        xt --> cand
        xt --> og
        ht_prev --> fg
        ht_prev --> ig
        ht_prev --> cand
        ht_prev --> og
        ct_prev --> ct
        fg -->|"× (element-wise)"| ct
        ig -->|"× candidate"| ct
        ct --> ht
        og -->|"× tanh(c(t))"| ht
    end
    
    style LSTMCell fill:#0f0f23,stroke:#ec4899,color:#fff
```

---

## ⚙️ Custom Components

### 1. AttentionLayer — Custom Keras Layer

**File**: `custom_components.py` (line 21–78)

Attention mechanism memungkinkan model memberikan **bobot berbeda** pada setiap timestep:

```mermaid
graph TB
    subgraph Attention["⭐ Attention Mechanism"]
        Input["Input: (batch, 3, 32)<br/>3 timesteps dari LSTM"]
        
        Score["Score = tanh(Input × W + b)<br/>Shape: (batch, 3, 32)"]
        
        Weights["Attention Weights = softmax(Score · u)<br/>Shape: (batch, 3, 1)<br/>Probabilitas per timestep"]
        
        Context["Context Vector = Σ(Input × Weights)<br/>Shape: (batch, 32)<br/>Weighted sum"]
    end
    
    Input --> Score
    Score --> Weights
    Weights --> Context
    
    Context --> Dense["→ Dense(16) → Output(1)"]
    
    style Attention fill:#1a1a2e,stroke:#f59e0b,color:#fff
```

**Intuisi**: Jika siklus ke-3 (terbaru) lebih mirip dengan pola historis, attention layer akan memberi bobot lebih tinggi ke siklus tersebut. Siklus yang anomali akan diberi bobot rendah.

```python
class AttentionLayer(keras.layers.Layer):
    def __init__(self, units=32):
        # W: weight matrix (feature_dim → units)
        # b: bias vector (units)
        # u: context vector (units)
    
    def call(self, inputs):
        # 1. score = tanh(inputs @ W + b)         → (batch, timesteps, units)
        # 2. weights = softmax(sum(score * u))     → (batch, timesteps, 1)
        # 3. context = sum(inputs * weights)       → (batch, features)
        return context_vector
```

### 2. CyclePredictionLoss — Custom Loss Function

**File**: `custom_components.py` (line 84–124)

Mengkombinasikan **Huber Loss** (robust terhadap outlier) dengan **Range Penalty**:

```mermaid
graph LR
    subgraph Loss["Custom Loss Function"]
        direction TB
        
        Input["y_true, y_pred"]
        
        Huber["📐 Huber Loss<br/>Smooth transition antara<br/>L1 (MAE) dan L2 (MSE)<br/>delta = 1.0"]
        
        Range["🚧 Range Penalty<br/>Penalti ekstra jika prediksi<br/>di luar range [0, 1]<br/>(scaled domain)"]
        
        Total["Total Loss = Huber + 0.1 × Range Penalty"]
    end
    
    Input --> Huber
    Input --> Range
    Huber --> Total
    Range --> Total
    
    style Loss fill:#1a1a2e,stroke:#ef4444,color:#fff
```

**Mengapa bukan MSE biasa?**

| Loss | Kelebihan | Kekurangan |
|------|-----------|-----------|
| MSE | Umum digunakan | Sensitif terhadap outlier |
| MAE | Robust terhadap outlier | Tidak smooth di 0 |
| **Huber (Kita)** | ✅ Robust + smooth | Sedikit lebih kompleks |
| **+ Range Penalty** | ✅ Mencegah prediksi absurd | Custom implementation |

### 3. TrainingMonitorCallback — Custom Callback

**File**: `custom_components.py` (line 130–229)

```mermaid
sequenceDiagram
    participant T as 🏋️ Training Loop
    participant CB as 📊 TrainingMonitor
    participant CSV as 📄 training_log.csv

    CB->>CB: on_train_begin → Print header, create log dir
    
    loop Every Epoch
        T->>CB: on_epoch_end(epoch, logs)
        CB->>CB: Track train_loss, val_loss, MAE, LR
        CB->>CB: Compare val_loss vs best_val_loss
        
        alt Improved
            CB->>CB: Update best_val_loss, reset wait counter
            CB->>CB: Print "📈 IMPROVED by X"
        else No Improvement
            CB->>CB: Increment wait counter
            CB->>CB: Print "⏳ No improvement (N/patience)"
        end
        
        CB->>CB: Append to training_log[]
    end
    
    CB->>CB: on_train_end → Print summary
    CB->>CSV: Export training_log to CSV
    
    Note over CB: Summary: Total epochs, best val_loss,<br/>best epoch, loss improvement, final LR
```

---

## 🏋️ Training Pipeline

### End-to-End Training Flow

```mermaid
graph TB
    subgraph Training["🏋️ Training Pipeline (train.py)"]
        direction TB
        
        Data["📊 Load Data<br/>data/sample_data.csv<br/>95 records, 8 users"]
        
        Preprocess["🔄 Preprocessing<br/>preprocess_for_training()"]
        
        subgraph PreprocessSteps["Preprocessing Steps"]
            Extract["Extract 5 features + target"]
            Scale["MinMaxScaler fit_transform"]
            Window["Sliding window (seq_len=3)"]
            Split["Train/Test split (80/20)"]
        end
        
        Build["🏗️ Build Model<br/>Functional API"]
        
        subgraph ModelLayers["Model Architecture"]
            L1["Input(3,5) → LSTM(64) → Dropout"]
            L2["→ LSTM(32) → Dropout"]
            L3["→ Attention(32) → Dense(16) → Output(1)"]
        end
        
        Compile["⚙️ Compile<br/>loss: CyclePredictionLoss<br/>optimizer: Adam(0.001)<br/>metrics: MAE"]
        
        Callbacks["📋 Callbacks"]
        
        subgraph CallbackList["Active Callbacks"]
            ES["EarlyStopping<br/>patience=20<br/>restore_best_weights"]
            RL["ReduceLROnPlateau<br/>factor=0.5<br/>patience=10<br/>min_lr=1e-6"]
            TB["TensorBoard<br/>log_dir=logs/tensorboard"]
            TM["TrainingMonitor<br/>(Custom Callback)"]
        end
        
        Fit["🚀 model.fit()<br/>epochs=200, batch_size=8<br/>validation_split=0.2"]
        
        Save["💾 Save Artifacts<br/>model/lstm_model.h5<br/>model/lstm_model.keras"]
        
        Eval["📊 Evaluate<br/>MSE, MAE on test set"]
    end
    
    Data --> Preprocess
    Preprocess --> PreprocessSteps
    PreprocessSteps --> Build
    Build --> ModelLayers
    ModelLayers --> Compile
    Compile --> Callbacks
    Callbacks --> CallbackList
    CallbackList --> Fit
    Fit --> Save
    Save --> Eval
    
    style Training fill:#0f0f23,stroke:#10b981,color:#fff
    style PreprocessSteps fill:#1a1a2e,stroke:#6366f1,color:#fff
    style ModelLayers fill:#1a1a2e,stroke:#ec4899,color:#fff
    style CallbackList fill:#1a1a2e,stroke:#f59e0b,color:#fff
```

### Training Configuration

| Parameter | Value | Alasan |
|-----------|-------|--------|
| **Epochs** | 200 (max) | EarlyStopping akan hentikan lebih awal |
| **Batch Size** | 8 | Dataset kecil → batch kecil |
| **Optimizer** | Adam (lr=0.001) | Adaptive learning rate, konvergen cepat |
| **Loss** | Custom Huber + Range Penalty | Robust terhadap outlier + range enforcement |
| **EarlyStopping** | patience=20, restore_best | Mencegah overfitting |
| **ReduceLROnPlateau** | factor=0.5, patience=10 | Fine-tune saat plateau |
| **Min Learning Rate** | 1e-6 | Batas bawah LR reduction |
| **Validation Split** | 20% | Monitor generalization |

### Learning Rate Schedule

```mermaid
graph LR
    Start["LR = 0.001"] -->|"10 epochs no improvement"| R1["LR = 0.0005"]
    R1 -->|"10 epochs no improvement"| R2["LR = 0.00025"]
    R2 -->|"10 epochs no improvement"| R3["LR = 0.000125"]
    R3 -->|"..."| Min["LR = 0.000001<br/>(min_lr)"]
    
    style Start fill:#ec4899,stroke:#fff,color:#fff
    style Min fill:#10b981,stroke:#fff,color:#fff
```

---

## 🔄 Preprocessing Pipeline

### Training Preprocessing

```mermaid
graph LR
    CSV["📄 CSV Data<br/>95 rows × 8 cols"] --> Validate["✅ Validate<br/>Required columns"]
    Validate --> Extract["📐 Extract<br/>5 features + 1 target"]
    Extract --> Scale["📏 MinMaxScaler<br/>fit_transform<br/>Range: [0, 1]"]
    Scale --> Window["🔲 Sliding Window<br/>seq_length = 3<br/>Creates sequences"]
    Window --> Split["✂️ Train/Test<br/>80% / 20%"]
    Split --> SaveScalers["💾 Save Scalers<br/>scaler_X.pkl<br/>scaler_y.pkl"]
    
    style CSV fill:#6366f1,stroke:#fff,color:#fff
    style Scale fill:#ec4899,stroke:#fff,color:#fff
    style Window fill:#a855f7,stroke:#fff,color:#fff
```

### Prediction Preprocessing

```mermaid
graph LR
    Input["📥 API Input<br/>cycles, sleep,<br/>stress, fasting"] --> Pad["📐 Pad/Trim<br/>to 3 timesteps"]
    Pad --> Estimate["📊 Estimate<br/>period_length<br/>(if missing)"]
    Estimate --> LoadScale["📏 Load Scalers<br/>scaler_X.pkl<br/>transform only"]
    LoadScale --> Reshape["🔲 Reshape<br/>(1, 3, 5)<br/>3D tensor"]
    Reshape --> Predict["🧠 Model<br/>Predict"]
    Predict --> Inverse["📐 Inverse<br/>Scale"]
    Inverse --> Clamp["🔒 Clamp<br/>21–45 days"]
    
    style Input fill:#3b82f6,stroke:#fff,color:#fff
    style Predict fill:#ec4899,stroke:#fff,color:#fff
    style Clamp fill:#10b981,stroke:#fff,color:#fff
```

### Feature Details

| # | Feature | Range | Scaling | Description |
|---|---------|-------|---------|-------------|
| 1 | `cycle_length` | 21–45 hari | MinMax [0,1] | Panjang siklus |
| 2 | `period_length` | 3–7 hari | MinMax [0,1] | Durasi menstruasi |
| 3 | `avg_sleep` | 1–5 | MinMax [0,1] | Kualitas tidur |
| 4 | `avg_stress` | 1–5 | MinMax [0,1] | Tingkat stres |
| 5 | `fasting_days` | 0+ | MinMax [0,1] | Hari puasa |

---

## 🚀 Inference & Deployment

### Flask API Server (`app.py`)

```mermaid
graph TB
    subgraph FlaskApp["🚀 Flask Application"]
        direction TB
        
        Init["App Initialization"]
        
        subgraph Startup["On Startup"]
            LoadModel["Load lstm_model.h5<br/>atau lstm_model.keras"]
            LoadScalers["Load scaler_X.pkl<br/>dan scaler_y.pkl"]
            SetGlobal["Set global model variable"]
        end
        
        subgraph Endpoints["API Endpoints"]
            Health["GET /health<br/>{ status, model_loaded }"]
            Predict["POST /predict<br/>Main prediction endpoint"]
        end
        
        subgraph PredictFlow["Prediction Processing"]
            Validate["Validate input"]
            PreProc["preprocess_for_prediction()"]
            Infer["model.predict()"]
            PostProc["inverse_transform + clamp"]
            Confidence["Calculate confidence"]
            Return["Return JSON response"]
        end
    end
    
    Init --> Startup
    Startup --> Endpoints
    Predict --> PredictFlow
    
    style FlaskApp fill:#0f0f23,stroke:#10b981,color:#fff
    style Startup fill:#1a1a2e,stroke:#a855f7,color:#fff
    style Endpoints fill:#1a1a2e,stroke:#3b82f6,color:#fff
    style PredictFlow fill:#1a1a2e,stroke:#ec4899,color:#fff
```

### Model Loading Strategy

```python
# Priority: .keras → .h5 → fallback
try:
    model = keras.models.load_model('model/lstm_model.keras', custom_objects={...})
except:
    try:
        model = keras.models.load_model('model/lstm_model.h5', custom_objects={...})
    except:
        model = None  # Fallback mode: use simple average
```

---

## 📊 Model Evaluation

### Metrics

| Metric | Training Set | Test Set | Target |
|--------|-------------|----------|--------|
| **MSE** (Mean Squared Error) | ~0.005 | ~0.008 | < 0.01 |
| **MAE** (Mean Absolute Error) | ~1.2 hari | ~1.8 hari | ≤ 2 hari ✅ |
| **Loss** (Custom Huber) | ~0.003 | ~0.006 | Converged |

### Training Curves (Typical)

```
Epoch   1: Loss: 0.15000 | Val Loss: 0.12000 | 📈 Initial
Epoch  20: Loss: 0.02500 | Val Loss: 0.03000 | 📈 Rapid improvement
Epoch  50: Loss: 0.00800 | Val Loss: 0.01200 | 📈 Steady progress
Epoch 100: Loss: 0.00350 | Val Loss: 0.00650 | ⏳ Plateau starting
Epoch 120: Loss: 0.00300 | Val Loss: 0.00600 | 🏁 EarlyStopping triggered
```

---

## 🔧 Hyperparameter Tuning

### Parameter Search Space

| Hyperparameter | Tested Values | Selected | Reasoning |
|----------------|--------------|----------|-----------|
| LSTM Units (L1) | 32, 64, 128 | **64** | Balance capacity vs overfitting |
| LSTM Units (L2) | 16, 32, 64 | **32** | Dimensionality reduction |
| Attention Units | 16, 32 | **32** | Match LSTM output dim |
| Dense Units | 8, 16, 32 | **16** | Sufficient for 1 output |
| Dropout Rate | 0.1, 0.2, 0.3 | **0.2** | Moderate regularization |
| Batch Size | 4, 8, 16 | **8** | Small dataset optimization |
| Sequence Length | 2, 3, 5 | **3** | Balance history vs data availability |
| Huber Delta | 0.5, 1.0, 2.0 | **1.0** | Standard transition point |

---

## 📂 File Reference

| File | Lines | Purpose |
|------|-------|---------|
| `train.py` | 210 | Model building, training, evaluation |
| `preprocess.py` | 200 | Data loading, scaling, sequencing |
| `custom_components.py` | 229 | AttentionLayer, CyclePredictionLoss, TrainingMonitor |
| `app.py` | 160 | Flask API server |

---

<div align="center">

[← Kembali ke Overview](README.md) · [Data Science →](README_DATA_SCIENCE.md)

**Dibuat oleh Ridho dan teman-teman — Capstone Project Coding Camp 2026**

**Powered by [kamidukung.biz.id](https://kamidukung.biz.id/)**

</div>
]]>
