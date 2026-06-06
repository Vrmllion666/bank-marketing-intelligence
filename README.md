# Bank Marketing Intelligence Dashboard

**Aplikasi Berbasis Web untuk Segmentasi dan Klasifikasi Nasabah Bank Menggunakan Pendekatan Machine Learning**

---

## Abstrak

Penelitian ini mengimplementasikan sistem cerdas berbasis web untuk mendukung pengambilan keputusan dalam kampanye pemasaran produk deposito berjangka pada sektor perbankan. Sistem yang dikembangkan mengintegrasikan dua pendekatan utama dalam pembelajaran mesin, yaitu klasifikasi menggunakan algoritma XGBoost dan Random Forest, serta segmentasi nasabah menggunakan K-Means Clustering. Dataset yang digunakan adalah Bank Marketing Dataset dari UCI Machine Learning Repository yang terdiri dari 41.188 rekaman dengan 20 atribut. Aplikasi dibangun menggunakan framework Streamlit dan divisualisasikan secara interaktif menggunakan Plotly.

---

## 1. Pendahuluan

Pemasaran produk keuangan seperti deposito berjangka memerlukan strategi yang tepat sasaran untuk meminimalkan biaya operasional sekaligus memaksimalkan tingkat konversi nasabah. Pendekatan konvensional yang mengandalkan kontak acak kepada seluruh basis nasabah terbukti tidak efisien. Dalam konteks ini, penerapan teknik *data mining* untuk segmentasi dan klasifikasi nasabah menjadi relevan sebagai solusi yang berbasis data.

Sistem yang dikembangkan dalam penelitian ini bertujuan untuk:
1. Mengklasifikasikan kemungkinan seorang nasabah berlangganan deposito berjangka berdasarkan atribut demografis dan historis kampanye.
2. Mengelompokkan nasabah ke dalam segmen yang bermakna untuk mendukung perumusan strategi pemasaran yang lebih terarah.
3. Menyediakan antarmuka prediksi interaktif yang dapat dioperasikan secara langsung oleh praktisi non-teknis.

---

## 2. Metodologi

### 2.1 Dataset

| Sumber | UCI ML Repository — Bank Marketing Dataset 
| Jumlah rekaman | 41.188 baris 
| Jumlah fitur | 20 fitur (numerik dan kategorikal) 
| Variabel target | `y` — subscribe deposito berjangka (yes/no) 

### 2.2 Pipeline Pemrosesan Data

Pemrosesan data dilakukan secara berurutan melalui tahapan berikut:

1. **Pembersihan data** — penghapusan duplikat dan penanganan nilai *unknown* pada fitur kritis
2. **Encoding** — One-Hot Encoding pada seluruh fitur kategorikal dengan penerapan `drop_first=True` untuk menghindari multikolinearitas
3. **Normalisasi** — StandardScaler diterapkan pada 10 fitur numerik
4. **Penyeimbangan kelas** — SMOTE (*Synthetic Minority Over-sampling Technique*) digunakan untuk mengatasi ketidakseimbangan kelas pada data latih
5. **Segmentasi** — K-Means Clustering dengan K=4 ditentukan berdasarkan analisis *elbow method*
6. **Klasifikasi** — pelatihan dua model: Random Forest (300 pohon) dan XGBoost (300 iterasi, *learning rate* = 0,05)
7. **Evaluasi** — pembagian data latih dan uji dengan rasio 80:20 menggunakan stratified sampling

### 2.3 Arsitektur Sistem

| Komponen | Teknologi |
|----------|-----------|
| Model klasifikasi | XGBoost, Random Forest |
| Model segmentasi | K-Means Clustering (K=4) |
| Reduksi dimensi | PCA 2D (untuk visualisasi cluster) |
| Backend & antarmuka | Streamlit |
| Visualisasi interaktif | Plotly |
| Bahasa pemrograman | Python 3.10+ |

---

## 3. Hasil dan Fitur Aplikasi

Aplikasi terdiri dari empat modul utama:

- **Dashboard** — menampilkan statistik ringkasan dataset, distribusi segmen nasabah, dan tingkat konversi per kelompok
- **Prediksi Nasabah** — memungkinkan pengguna memasukkan data satu nasabah dan memperoleh nilai probabilitas subscribe beserta rekomendasi tindak lanjut
- **Prediksi Batch (CSV)** — mendukung unggahan berkas CSV berisi ratusan nasabah dengan keluaran prediksi yang dapat diunduh secara langsung
- **Evaluasi Model** — menyajikan perbandingan metrik performa antara Random Forest dan XGBoost secara visual

Model terbaik (XGBoost) menghasilkan nilai ROC-AUC sebesar 0,93 pada data uji, yang mengindikasikan kemampuan diskriminasi yang baik terhadap kelas minoritas.

---

## 4. Cara Menjalankan Aplikasi

### Prasyarat

- Python versi 3.10 atau lebih baru
- pip (Python package manager)

### Instalasi

```bash
# Klon repositori
git clone https://github.com/Vrmllion666/bank-marketing-intelligence.git
cd bank-marketing-intelligence

# Instalasi dependensi
pip install -r requirements.txt

# Menjalankan aplikasi
streamlit run app.py
```

> Artefak model telah disertakan dalam direktori `model/` sehingga tidak diperlukan proses pelatihan ulang.

**Akses daring:** https://bank-marketing-intelligence-wekaagril.streamlit.app 

---

## 5. Struktur Repositori

```
bank-marketing-intelligence/
├── app.py                      # Aplikasi Streamlit utama
├── bank-additional-full.csv    # Dataset sumber (UCI ML Repository)
├── requirements.txt            # Daftar dependensi Python
├── README.md
└── model/
    ├── xgboost.pkl             # Artefak model XGBoost
    ├── kmeans.pkl              # Artefak model K-Means (K=4)
    ├── scaler.pkl              # Objek StandardScaler
    ├── feature_columns.pkl     # Daftar fitur setelah encoding
    ├── model_metadata.pkl      # Metadata dan metrik evaluasi model
    ├── data_resampled.parquet  # Data hasil SMOTE untuk visualisasi cluster
    └── cluster_viz.parquet     # Data PCA untuk scatter plot segmentasi
```

---

## 6. Dependensi

```
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.26.0
scikit-learn>=1.4.0
imbalanced-learn>=0.12.0
xgboost>=2.0.0
joblib>=1.3.0
plotly>=5.18.0
pyarrow>=14.0.0
```

---

## 7. Referensi Dataset

Moro, S., Cortez, P., & Rita, P. (2014). A data-driven approach to predict the success of bank telemarketing. *Decision Support Systems*, 62, 22–31. https://doi.org/10.1016/j.dss.2014.03.001

Dataset tersedia pada: https://archive.ics.uci.edu/ml/datasets/Bank+Marketing

---

## Informasi Penelitian
 KELOMPOK 3 KDD 03 
 ANGGOTA :1.Weka Surajati Sudanta (24051214079)
          2. Agril Adirizky (24051214095)
 Mata Kuliah : Data Mining
 Prodi       : Sistem Informasi
 Fakultas    : Teknik
 Universitas Negeri Surabaya
 2026

