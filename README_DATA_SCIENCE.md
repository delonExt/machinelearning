<div align="center">

# 📊 Data Science — YeoCycles ML Service

### Exploratory Data Analysis, Statistical Testing & Interactive Dashboard

[← Kembali ke Overview](README.md) · [AI Engineering →](README_AI_ENGINEERING.md)

</div>

---

## 📑 Daftar Isi

- [Pipeline Overview](#-data-science-pipeline)
- [Dataset](#-dataset)
- [Data Gathering](#-phase-1-data-gathering)
- [Data Assessing](#-phase-2-data-assessing)
- [Data Cleaning](#-phase-3-data-cleaning)
- [Feature Engineering](#️-phase-4-feature-engineering)
- [EDA & Visualisasi](#-phase-5-exploratory-data-analysis)
- [Explanatory Analysis](#-phase-6-explanatory-analysis)
- [Streamlit Dashboard](#-streamlit-interactive-dashboard)
- [Business Questions](#-menjawab-pertanyaan-bisnis)
- [Kesimpulan & Insight](#-kesimpulan--insight)

---

## 🗺️ Data Science Pipeline

### End-to-End Pipeline Overview

```mermaid
graph LR
    subgraph Phase1["📥 Phase 1"]
        Gather["Data<br/>Gathering<br/>CSV Load"]
    end
    
    subgraph Phase2["🔍 Phase 2"]
        Assess["Data<br/>Assessing<br/>Quality Check"]
    end
    
    subgraph Phase3["🧹 Phase 3"]
        Clean["Data<br/>Cleaning<br/>Validation"]
    end
    
    subgraph Phase4["⚙️ Phase 4"]
        FE["Feature<br/>Engineering<br/>3 New Features"]
    end
    
    subgraph Phase5["📊 Phase 5"]
        EDA["EDA<br/>11 Visualisasi<br/>Statistik"]
    end
    
    subgraph Phase6["💡 Phase 6"]
        Explain["Explanatory<br/>Analysis<br/>BQ2-BQ5"]
    end
    
    subgraph Outputs["📈 Outputs"]
        Plots["11 PNG<br/>Visualisasi"]
        Dashboard["Streamlit<br/>Dashboard"]
        Insights["Insight &<br/>Kesimpulan"]
    end
    
    Gather --> Assess --> Clean --> FE --> EDA --> Explain
    Explain --> Plots
    Explain --> Dashboard
    Explain --> Insights
    
    style Phase1 fill:#6366f1,stroke:#fff,color:#fff
    style Phase2 fill:#8b5cf6,stroke:#fff,color:#fff
    style Phase3 fill:#a855f7,stroke:#fff,color:#fff
    style Phase4 fill:#d946ef,stroke:#fff,color:#fff
    style Phase5 fill:#ec4899,stroke:#fff,color:#fff
    style Phase6 fill:#f43f5e,stroke:#fff,color:#fff
    style Outputs fill:#10b981,stroke:#fff,color:#fff
```

### Tools & Libraries

| Tool | Purpose |
|------|---------|
| **Pandas** | Data manipulation & analysis |
| **NumPy** | Numerical computing |
| **Matplotlib** | Static visualisasi |
| **Seaborn** | Statistical visualisasi |
| **SciPy** | Statistical tests (Pearson, Shapiro-Wilk) |
| **Plotly** | Interactive visualisasi (Streamlit) |
| **Streamlit** | Interactive dashboard web app |

---

## 📋 Dataset

### Dataset Overview

| Item | Detail |
|------|--------|
| **Nama** | Menstrual Cycle Tracking Dataset |
| **Sumber** | Data sintetis berbasis pola medis |
| **Records** | 95 records |
| **Users** | 8 pengguna unik |
| **Periode** | Januari 2025 – Desember 2025 |
| **Format** | CSV |
| **Lokasi** | `data/sample_data.csv` |

### Kolom Dataset

```mermaid
graph TB
    subgraph RawFeatures["📊 Raw Features (7 kolom)"]
        uid["user_id<br/>INT, 1-8"]
        cs["cycle_start<br/>DATE"]
        cl["cycle_length<br/>INT, 24-35 hari"]
        pl["period_length<br/>INT, 4-7 hari"]
        as2["avg_sleep<br/>FLOAT, 5.0-8.5"]
        ast["avg_stress<br/>FLOAT, 1.0-5.0"]
        fd["fasting_days<br/>INT, 0-3"]
    end
    
    subgraph Target["🎯 Target Variable"]
        ncl["next_cycle_length<br/>INT, 24-35 hari<br/>YANG DIPREDIKSI"]
    end
    
    subgraph Engineered["⚙️ Engineered Features (3 kolom)"]
        cr["cycle_regularity<br/>|cycle - mean_user|"]
        ssr["sleep_stress_ratio<br/>sleep / (stress + 0.1)"]
        ir["is_irregular<br/>1 if cycle < 21 or > 35"]
    end
    
    cl --> ncl
    pl --> ncl
    as2 --> ncl
    ast --> ncl
    fd --> ncl
    
    cl --> cr
    as2 --> ssr
    ast --> ssr
    cl --> ir
    
    style RawFeatures fill:#1a1a2e,stroke:#6366f1,color:#fff
    style Target fill:#1a1a2e,stroke:#ec4899,color:#fff
    style Engineered fill:#1a1a2e,stroke:#10b981,color:#fff
```

### Statistik Deskriptif

| Variabel | Mean | Std | Min | Max | Median |
|----------|------|-----|-----|-----|--------|
| cycle_length | 28.6 | 3.1 | 24 | 35 | 28 |
| period_length | 5.2 | 1.1 | 4 | 7 | 5 |
| avg_sleep | 6.8 | 1.0 | 5.0 | 8.5 | 7.0 |
| avg_stress | 2.8 | 1.1 | 1.0 | 5.0 | 2.8 |
| fasting_days | 0.7 | 1.0 | 0 | 3 | 0 |
| next_cycle_length | 28.7 | 3.0 | 24 | 35 | 28 |

---

## 📥 Phase 1: Data Gathering

**Script**: `eda_analysis.py` → `data_gathering()`

```python
df = pd.read_csv('data/sample_data.csv')
# → 95 rows × 8 columns
# → 8 pengguna unik, masing-masing 10-12 records
```

### Data Quality Summary

| Check | Status | Detail |
|-------|--------|--------|
| Missing values | ✅ 0 | Semua kolom terisi lengkap |
| Duplicate rows | ✅ 0 | Setiap record unik |
| Data types | ✅ Konsisten | Numerik sesuai tipe |
| Range validity | ✅ Valid | Semua dalam range medis |
| Data source | ⚠️ Sintetis | Berbasis pola medis, bukan pasien nyata |

---

## 🔍 Phase 2: Data Assessing

**Script**: `eda_analysis.py` → `data_assessing()`

### Outlier Detection (IQR Method)

```mermaid
graph LR
    subgraph IQR["IQR Method per Variabel"]
        Q1["Q1 = 25th percentile"]
        Q3["Q3 = 75th percentile"]
        IQRCalc["IQR = Q3 - Q1"]
        Lower["Lower = Q1 - 1.5 × IQR"]
        Upper["Upper = Q3 + 1.5 × IQR"]
    end
    
    subgraph Results["Hasil"]
        CL["cycle_length: ✅ No outliers"]
        PL["period_length: ✅ No outliers"]
        AS["avg_sleep: ✅ No outliers"]
        AST["avg_stress: ✅ No outliers"]
        FD["fasting_days: ✅ No outliers"]
    end
    
    IQR --> Results
    
    style IQR fill:#1a1a2e,stroke:#f59e0b,color:#fff
    style Results fill:#1a1a2e,stroke:#10b981,color:#fff
```

---

## 🧹 Phase 3: Data Cleaning

**Script**: `eda_analysis.py` → `data_cleaning()`

### Proses Cleaning

```mermaid
graph TB
    Raw["📄 Raw Data<br/>95 rows × 8 cols"] --> TypeConvert["🔄 Type Conversion<br/>cycle_start → datetime"]
    TypeConvert --> Validate["✅ Range Validation<br/>cycle: 21-45, period: 1-10<br/>sleep: 0-12, stress: 0-5"]
    Validate --> FE["⚙️ Feature Engineering<br/>(Phase 4)"]
    FE --> Clean["✨ Clean Data<br/>95 rows × 12 cols"]
    
    style Raw fill:#ef4444,stroke:#fff,color:#fff
    style Clean fill:#10b981,stroke:#fff,color:#fff
```

---

## ⚙️ Phase 4: Feature Engineering

### 3 Fitur Baru yang Dibuat

```mermaid
graph TB
    subgraph FE["⚙️ Feature Engineering"]
        direction LR
        
        subgraph F1["cycle_regularity"]
            Formula1["= |cycle_length - mean_per_user|"]
            Desc1["Mengukur deviasi dari<br/>rata-rata siklus individu.<br/>Tinggi = tidak teratur."]
        end
        
        subgraph F2["sleep_stress_ratio"]
            Formula2["= avg_sleep / (avg_stress + 0.1)"]
            Desc2["Rasio kualitas hidup.<br/>Tinggi = kondisi baik<br/>(tidur cukup, stres rendah)."]
        end
        
        subgraph F3["is_irregular"]
            Formula3["= 1 if cycle < 21 or > 35"]
            Desc3["Flag biner WHO.<br/>Menandai siklus<br/>di luar range normal."]
        end
    end
    
    style F1 fill:#ec4899,stroke:#fff,color:#fff
    style F2 fill:#a855f7,stroke:#fff,color:#fff
    style F3 fill:#6366f1,stroke:#fff,color:#fff
```

| Fitur Baru | Formula | Tujuan |
|-----------|---------|--------|
| `cycle_regularity` | `\|cycle_length - mean_user\|` | Ukur ketidakteraturan individu |
| `sleep_stress_ratio` | `sleep / (stress + 0.1)` | Indikator kualitas hidup |
| `is_irregular` | `1 if cycle < 21 or > 35` | Flag WHO standar |

---

## 📊 Phase 5: Exploratory Data Analysis

**Script**: `eda_analysis.py` → `eda_analysis()`

### 11 Visualisasi yang Dihasilkan

| # | Visualisasi | File | Insight |
|---|-------------|------|---------|
| 01 | Distribusi Cycle Length | `01_distribusi_cycle_length.png` | Mean ~28.6 hari, distribusi mendekati normal |
| 02 | Boxplot per User | `02_boxplot_per_user.png` | Variasi antar user terlihat jelas |
| 03 | Correlation Heatmap | `03_correlation_heatmap.png` | cycle→next_cycle korelasi kuat (r≈0.85) |
| 04 | Scatter Plots | `04_scatter_plots.png` | Stres positif, tidur negatif thd siklus |
| 05 | Distribusi Semua Variabel | `05_distribusi_semua_variabel.png` | Overview distribusi 6 variabel |
| 06 | Tren per User | `06_tren_per_user.png` | Pola individual yang konsisten |
| 07 | Pairplot | `07_pairplot.png` | Hubungan multivariat |
| 08 | BQ2: Lifestyle Impact | `08_bq2_lifestyle_impact.png` | Stres signifikan, puasa tidak |
| 09 | BQ3: Variasi per User | `09_bq3_variasi_per_user.png` | Rata-rata ± SD per pengguna |
| 10 | BQ4: Regularitas | `10_bq4_regularitas.png` | Mayoritas CV < 10% (regular) |
| 11 | BQ5: Distribusi Overall | `11_bq5_distribusi_keseluruhan.png` | QQ-Plot dan KDE analysis |

### Correlation Analysis

```mermaid
graph TB
    subgraph Correlations["🔗 Korelasi dengan Target (next_cycle_length)"]
        direction LR
        
        Strong["🟢 KUAT (r > 0.5)"]
        Moderate["🟡 MODERAT (0.2 < r < 0.5)"]
        Weak["🔴 LEMAH (r < 0.2)"]
        
        CL["cycle_length<br/>r ≈ +0.85<br/>SANGAT KUAT ✅"]
        SSR["sleep_stress_ratio<br/>r ≈ -0.35"]
        AS["avg_stress<br/>r ≈ +0.25"]
        SL["avg_sleep<br/>r ≈ -0.20"]
        FD["fasting_days<br/>r ≈ +0.05<br/>TIDAK SIGNIFIKAN"]
    end
    
    Strong --> CL
    Moderate --> SSR
    Moderate --> AS
    Weak --> SL
    Weak --> FD
    
    style Strong fill:#10b981,stroke:#fff,color:#fff
    style Moderate fill:#f59e0b,stroke:#fff,color:#fff
    style Weak fill:#ef4444,stroke:#fff,color:#fff
```

### Key EDA Findings

```mermaid
graph TB
    subgraph Findings["📊 Temuan Utama EDA"]
        F1["1️⃣ Cycle length 24-35 hari<br/>Mean = 28.6 hari<br/>(sesuai WHO range normal)"]
        
        F2["2️⃣ Korelasi KUAT antara<br/>current cycle → next cycle<br/>(r ≈ 0.85+)<br/>→ Mendukung LSTM approach"]
        
        F3["3️⃣ Stres berkorelasi POSITIF<br/>dengan panjang siklus<br/>(semakin stres → siklus lebih panjang)"]
        
        F4["4️⃣ Tidur berkorelasi NEGATIF<br/>dengan panjang siklus<br/>(tidur baik → siklus lebih regular)"]
        
        F5["5️⃣ Puasa: pengaruh MINIMAL<br/>Tidak signifikan secara statistik"]
        
        F6["6️⃣ Mayoritas user REGULAR<br/>(CV < 10%)<br/>Namun variasi individual signifikan"]
    end
    
    style Findings fill:#0f0f23,stroke:#a855f7,color:#fff
```

---

## 💡 Phase 6: Explanatory Analysis

**Script**: `eda_analysis.py` → `explanatory_analysis()`

### Menjawab Business Questions

#### BQ1: Seberapa akurat model LSTM?

| Metrik | Hasil | Target | Status |
|--------|-------|--------|--------|
| **MAE** | ~1.8 hari | ≤ 2 hari | ✅ Tercapai |
| **MSE** | ~0.008 | < 0.01 | ✅ Tercapai |

#### BQ2: Apakah faktor gaya hidup berpengaruh?

```mermaid
graph LR
    subgraph BQ2["💡 BQ2: Pengaruh Gaya Hidup"]
        direction TB
        
        Sleep["😴 avg_sleep<br/>r = -0.20<br/>p < 0.05<br/>✅ SIGNIFIKAN"]
        
        Stress["😰 avg_stress<br/>r = +0.25<br/>p < 0.05<br/>✅ SIGNIFIKAN"]
        
        Fast["🍽️ fasting_days<br/>r = +0.05<br/>p > 0.05<br/>❌ TIDAK signifikan"]
    end
    
    Sleep --> Conclusion["📝 Kesimpulan:<br/>Stres & tidur berpengaruh<br/>Puasa tidak berpengaruh"]
    Stress --> Conclusion
    Fast --> Conclusion
    
    style Sleep fill:#3b82f6,stroke:#fff,color:#fff
    style Stress fill:#ef4444,stroke:#fff,color:#fff
    style Fast fill:#6b7280,stroke:#fff,color:#fff
    style Conclusion fill:#10b981,stroke:#fff,color:#fff
```

#### BQ3: Variasi antar pengguna?

| User | Mean Cycle | Std Dev | Range | Status |
|------|-----------|---------|-------|--------|
| User 1 | 27.5 | 1.8 | 25-30 | 🟢 Regular |
| User 2 | 29.0 | 2.1 | 26-33 | 🟢 Regular |
| User 3 | 30.2 | 3.5 | 24-35 | 🟡 Moderate |
| User 4 | 27.8 | 1.5 | 25-30 | 🟢 Regular |
| User 5 | 28.5 | 2.3 | 25-32 | 🟢 Regular |
| User 6 | 29.3 | 2.8 | 25-34 | 🟡 Moderate |
| User 7 | 28.0 | 1.9 | 25-31 | 🟢 Regular |
| User 8 | 29.5 | 2.5 | 26-34 | 🟢 Regular |

#### BQ4: Regular atau irregular?

```mermaid
pie title Distribusi Regularitas Siklus (CV%)
    "🟢 Regular (CV < 5%)" : 4
    "🟡 Moderate (5% < CV < 10%)" : 3
    "🔴 Irregular (CV > 10%)" : 1
```

#### BQ5: Distribusi keseluruhan?

| Statistik | Nilai | Interpretasi |
|-----------|-------|-------------|
| **Skewness** | ~0.2 | Slight right-skew (mendekati simetris) |
| **Kurtosis** | ~-0.5 | Platykurtic (distribusi agak datar) |
| **Shapiro-Wilk p** | > 0.05 | Mendekati normal |

---

## 📈 Streamlit Interactive Dashboard

**File**: `streamlit_app.py` · **Jalankan**: `streamlit run streamlit_app.py`

### Dashboard Pages

```mermaid
graph TB
    subgraph StreamlitApp["📈 Streamlit Dashboard"]
        direction TB
        
        Sidebar["🎛️ Sidebar<br/>Navigation + User Filter"]
        
        subgraph Pages["6 Halaman Interaktif"]
            P1["🏠 Overview<br/>5 metrics, statistik deskriptif,<br/>data quality check"]
            P2["📊 Distribusi Data<br/>Histogram, violin plot,<br/>variable selector"]
            P3["🔗 Korelasi<br/>Heatmap, bar chart,<br/>scatter matrix"]
            P4["👤 Analisis per User<br/>User stats table, trend line,<br/>deep-dive per user"]
            P5["💡 Business Questions<br/>BQ2-BQ5 visualisasi interaktif"]
            P6["📝 Kesimpulan<br/>7 temuan utama,<br/>rekomendasi, arsitektur"]
        end
    end
    
    Sidebar --> P1
    Sidebar --> P2
    Sidebar --> P3
    Sidebar --> P4
    Sidebar --> P5
    Sidebar --> P6
    
    style StreamlitApp fill:#0f0f23,stroke:#ec4899,color:#fff
    style Pages fill:#1a1a2e,stroke:#a855f7,color:#fff
```

### Fitur Dashboard

| Halaman | Komponen | Interaktivitas |
|---------|----------|----------------|
| **🏠 Overview** | 5 metric cards, descriptive stats table, data quality checks | User filter (multiselect) |
| **📊 Distribusi** | Plotly histogram + marginal box/violin | Variable selector dropdown |
| **🔗 Korelasi** | Heatmap (RdPu), bar chart, scatter matrix | Variable multiselect |
| **👤 Per User** | Stats table, trend lines, deep-dive | User selector |
| **💡 BQ** | Pearson tests, regplots, CV analysis | Interactive filters |
| **📝 Kesimpulan** | Temuan, rekomendasi, arsitektur | Static content |

### Teknologi Dashboard

| Teknologi | Purpose |
|-----------|---------|
| **Streamlit** | Web framework (auto-reload, reactive) |
| **Plotly Express** | Interactive charts (hover, zoom, pan) |
| **Plotly Graph Objects** | Custom chart layouts |
| **SciPy.stats** | Pearson correlation, Shapiro-Wilk test |
| **`@st.cache_data`** | Data caching untuk performance |

---

## 📝 Kesimpulan & Insight

### 7 Temuan Utama

```mermaid
graph TB
    subgraph Insights["💡 7 Temuan Utama"]
        I1["1️⃣ Panjang siklus 24-35 hari<br/>(rata-rata 28.6, sesuai WHO)"]
        I2["2️⃣ Korelasi KUAT cycle → next_cycle<br/>(r ≈ 0.85+, mendukung LSTM)"]
        I3["3️⃣ STRES berkorelasi POSITIF<br/>(siklus lebih panjang saat stres tinggi)"]
        I4["4️⃣ TIDUR berkorelasi NEGATIF<br/>(tidur baik → siklus lebih regular)"]
        I5["5️⃣ PUASA: pengaruh MINIMAL<br/>(tidak signifikan secara statistik)"]
        I6["6️⃣ Mayoritas pengguna REGULAR<br/>(CV < 10%), variasi individual ada"]
        I7["7️⃣ Data TIDAK sepenuhnya normal<br/>(multimodal karena variasi per user)"]
    end
    
    style Insights fill:#0f0f23,stroke:#10b981,color:#fff
```

### Rekomendasi

| # | Rekomendasi | Alasan |
|---|-------------|--------|
| 1 | **Model LSTM cocok** | Pola sekuensial yang kuat (r≈0.85) |
| 2 | **Feature engineering penting** | `cycle_regularity` dan `sleep_stress_ratio` informatif |
| 3 | **Personalisasi per user** | Lebih efektif dari model general |
| 4 | **Perlu lebih banyak data** | Min 6+ siklus per user untuk akurasi optimal |
| 5 | **Stres monitoring** | Faktor gaya hidup paling berpengaruh |

---

## 📂 File Reference

| File | Lines | Purpose |
|------|-------|---------|
| `eda_analysis.py` | 501 | Complete EDA pipeline (5 phases, 11 visualisasi) |
| `streamlit_app.py` | 420 | Interactive Streamlit dashboard (6 pages) |
| `data_dictionary.md` | 87 | Dataset column documentation |
| `problem_discovery.md` | 107 | Problem statement, methodology, architecture |
| `data/sample_data.csv` | 95 rows | Training dataset |
| `analysis_output/` | 11 files | EDA visualization outputs |

### Menjalankan Analysis

```bash
# EDA Pipeline (generates 11 PNG visualizations)
python eda_analysis.py
# → Output: analysis_output/*.png

# Interactive Dashboard
streamlit run streamlit_app.py
# → http://localhost:8501
```

---

<div align="center">

[← Kembali ke Overview](README.md) · [AI Engineering →](README_AI_ENGINEERING.md)

**Dibuat oleh Ridho dan teman-teman — Capstone Project Coding Camp 2026**

**Powered by [kamidukung.biz.id](https://kamidukung.biz.id/)**

</div>

