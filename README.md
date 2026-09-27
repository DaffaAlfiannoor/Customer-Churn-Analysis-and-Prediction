# Ecommerce Customer Churn Analysis and Prediction

Analisis end-to-end dan model machine learning untuk memprediksi *customer churn* pada dataset Telco Customer Churn — mencakup audit kualitas data, exploratory data analysis (EDA), analisis statistik, feature engineering, perbandingan model, tuning hyperparameter, optimasi threshold, interpretabilitas model, hingga segmentasi risiko pelanggan untuk kebutuhan retensi.

## Daftar Isi

- [Ringkasan Project](#ringkasan-project)
- [Dataset](#dataset)
- [Struktur Analisis](#struktur-analisis)
- [Metodologi](#metodologi)
- [Hasil Model](#hasil-model)
- [Temuan Utama](#temuan-utama)
- [Instalasi & Menjalankan Project](#instalasi--menjalankan-project)
- [Struktur Repository](#struktur-repository)
- [Keterbatasan](#keterbatasan)

## Ringkasan Project

Churn pelanggan adalah salah satu masalah bisnis paling mahal bagi perusahaan berbasis langganan. Project ini membangun model klasifikasi untuk memprediksi probabilitas seorang pelanggan akan berhenti berlangganan (churn), sekaligus menjelaskan **mengapa** pelanggan tersebut berisiko — sehingga tim retensi dapat memprioritaskan intervensi berdasarkan data, bukan intuisi.

Fokus project ini bukan hanya akurasi model, tetapi **proses analisis yang dapat dipertanggungjawabkan**: setiap keputusan pembersihan data, pemilihan fitur, pemilihan model, dan pemilihan threshold didokumentasikan beserta alasannya.

## Dataset

- **Sumber**: Telco Customer Churn (`Telco-Customer-Churn.csv`)
- **Jumlah baris**: 7.043 pelanggan
- **Target**: `Churn` (Yes/No)
- **Fitur**: 19 atribut pelanggan mencakup:
  - **Demografi**: gender, SeniorCitizen, Partner, Dependents
  - **Layanan**: PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
  - **Kontrak & tagihan**: Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges, tenure

**Distribusi target**: ~73,5% tidak churn vs ~26,5% churn (imbalanced ringan, rasio ±3:1).

## Struktur Analisis

Notebook (`prediction.ipynb`) disusun dalam 24 bagian berurutan:

| No | Bagian | Deskripsi |
|----|--------|-----------|
| 1–2 | Import & Overview | Import library dan pemeriksaan awal struktur dataset |
| 3 | Data Quality Audit | Audit menyeluruh: missing value tersembunyi, whitespace, kategori tidak konsisten, nilai mustahil |
| 4 | Exploratory Data Analysis | Analisis univariat dan bivariat terhadap churn |
| 5 | Count vs. Rate Analysis | Membedakan jumlah pelanggan vs. tingkat churn per segmen |
| 6 | Financial Analysis | Hubungan biaya (MonthlyCharges, TotalCharges) dengan churn |
| 7 | Multivariate Analysis | Interaksi antar variabel (Contract × InternetService, dsb.) |
| 8 | Statistical Analysis | Korelasi Pearson, uji Chi-Square, dan Cramer's V |
| 9 | Feature Engineering | Pembuatan fitur `total_services`, dengan justifikasi fitur yang tidak dibuat |
| 10 | ML Setup | Pipeline preprocessing bebas data leakage |
| 11 | Baseline Model | Logistic Regression sebagai tolok ukur |
| 12–14 | Model Comparison | Perbandingan 5 classifier dengan cross-validation |
| 15 | Hyperparameter Tuning | RandomizedSearchCV pada 2 model terbaik |
| 16 | Class Imbalance | Evaluasi class weighting vs. threshold adjustment |
| 17 | Threshold Optimization | Trade-off precision/recall di berbagai threshold |
| 18 | ROC & PR Curves | Evaluasi performa menyeluruh model final |
| 19 | Error Analysis | Investigasi pola false positive/false negative |
| 20 | Model Interpretability | Koefisien regresi + Permutation Importance |
| 21 | Customer Risk Segmentation | Segmentasi pelanggan Low/Medium/High risk |
| 22–24 | Business Insights, Limitations, Conclusion | Ringkasan temuan bisnis dan keterbatasan analisis |

## Metodologi

**Data Cleaning**
- `TotalCharges` tersimpan sebagai teks dengan 11 baris berisi whitespace kosong. Diidentifikasi bahwa seluruh baris ini memiliki `tenure == 0` (pelanggan baru), sehingga diisi dengan nilai 0 — bukan dihapus.
- Tidak ditemukan whitespace, inkonsistensi kategori, atau nilai negatif/mustahil lainnya.

**Feature Engineering**
- `total_services`: jumlah layanan tambahan (0–8) yang dilanggan pelanggan, meringkas 8 kolom biner menjadi satu sinyal.
- Fitur derivatif seperti `tenure_group` sengaja dikeluarkan dari input model karena redundan dengan `tenure`.

**Preprocessing Pipeline**
- `StandardScaler` untuk fitur numerik, `OneHotEncoder` untuk fitur kategorikal, dibungkus dalam `ColumnTransformer` + `Pipeline` scikit-learn agar bebas dari *data leakage* (fit hanya pada data latih, termasuk di setiap fold cross-validation).
- Split data: 80/20, stratified terhadap target.

**Model yang Dibandingkan**

| Model | Alasan Disertakan |
|---|---|
| Logistic Regression | Baseline interpretable, cepat |
| Decision Tree | Menguji non-linearitas & interaksi |
| Random Forest | Ensemble untuk stabilitas |
| Gradient Boosting | Model tabular yang biasanya kuat |
| HistGradientBoosting | Varian boosting berbasis histogram |

**Evaluasi**
- Metrik utama: ROC-AUC, Precision, Recall, F1 (akurasi tidak dijadikan acuan utama karena data imbalanced).
- 5-fold Stratified Cross-Validation untuk estimasi performa yang stabil.
- RandomizedSearchCV untuk tuning hyperparameter pada 2 model terbaik.
- Threshold optimization untuk menyesuaikan trade-off precision/recall dengan kebutuhan bisnis.

## Hasil Model

| Model | CV ROC-AUC | Keterangan |
|---|---|---|
| Gradient Boosting (tuned) | ~0.848 | Model terbaik, setara secara statistik dengan Logistic Regression |
| Logistic Regression (tuned) | ~0.846 | Hampir menyamai model boosting — mengindikasikan sinyal churn bersifat aditif |
| Random Forest | ~0.83–0.84 | Kompetitif |
| Decision Tree | ~0.824 | Model termuda di antara 5 kandidat |

Perbedaan performa antar model **lebih kecil dari variasi antar-fold**, sehingga pemilihan algoritma dinilai kurang berpengaruh dibandingkan kualitas fitur yang tersedia.

## Temuan Utama

**Faktor pendorong churn tertinggi:**
- Kontrak bulanan (*month-to-month*) — churn rate 42,7% vs. 2,8% pada kontrak 2 tahun
- Masa berlangganan (*tenure*) pendek, terutama di bawah 12 bulan (47,4%)
- Layanan internet fiber optic (41,9%)
- Metode pembayaran *electronic check*
- Tidak memiliki `OnlineSecurity` atau `TechSupport`

**Faktor yang TIDAK berasosiasi signifikan dengan churn:**
- `gender` (p ≈ 0,49)
- `PhoneService` (p ≈ 0,34)

**Fitur paling berpengaruh pada model** (berdasarkan koefisien & permutation importance): `Contract`, `tenure`, dan `InternetService`.

**Segmentasi risiko**: Pelanggan pada set uji dikelompokkan menjadi Low/Medium/High risk berdasarkan probabilitas prediksi, memungkinkan tim retensi memprioritaskan *outreach* pada segmen High-risk yang secara disproporsional menampung sebagian besar pelanggan yang benar-benar churn.

## Instalasi & Menjalankan Project

### Prasyarat
- Python 3.9+
- Jupyter Notebook / JupyterLab

### Instalasi Dependensi

```bash
pip install numpy pandas scipy matplotlib seaborn scikit-learn jupyter
```

### Menjalankan Notebook

```bash
git clone <repository-url>
cd <repository-folder>
jupyter notebook prediction.ipynb
```

Pastikan file `Telco-Customer-Churn.csv` berada di direktori yang sama dengan notebook sebelum menjalankan sel `pd.read_csv(...)`.

## Struktur Repository

```
.
├── prediction.ipynb           # Notebook analisis dan pemodelan end-to-end
├── Telco-Customer-Churn.csv   # Dataset (tidak disertakan, harus disiapkan terpisah)
└── README.md                  # Dokumentasi project ini
```

## Keterbatasan

- **Observasional, bukan eksperimental** — seluruh hubungan yang ditemukan bersifat asosiasi, bukan kausalitas.
- **Variabel kontekstual yang hilang** — tidak tersedia data tiket dukungan, keluhan, gangguan jaringan, atau penawaran kompetitor yang kemungkinan besar menjelaskan sisa error model.
- **Data cross-sectional** — setiap pelanggan hanya diobservasi sekali, sehingga *time-to-churn* dan tren musiman tidak dapat dimodelkan.
- **Concept drift** — performa model berpotensi menurun seiring waktu karena perubahan perilaku pelanggan, harga, dan kompetisi pasar; diperlukan monitoring berkala jika dipakai di produksi.
