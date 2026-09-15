# Laboratorium Data & Model Machine Learning: Prediksi Harga Rumah Kota Surabaya

Repositori ini merupakan modul eksperimen data dan kecerdasan buatan (Machine Learning) untuk penelitian skripsi:

> **Judul Skripsi**: *Prediksi Harga Rumah di Kota Surabaya Menggunakan Algoritma XGBoost Berbasis Feature Engineering*  
> **Penyusun**: Raditya Rahmatullah Mochamad Patria (NIM: 434231086)  
> **Program Studi**: D4 Teknik Informatika, Fakultas Vokasi, Universitas Airlangga (2026)  

---

## 📌 Ringkasan & Posisi Terhadap Skripsi

Modul ini berfokus pada **hulu penelitian** (Data Science & Machine Learning Pipeline).  
Tujuan utama modul ini adalah menjawab **Rumusan Masalah 1 dan 2 pada Bab 1**, serta memenuhi batasan metodologi pada **Bab 3 Subbab 3.3, 3.4, dan 3.5**:
1. **Mengumpulkan data riil** dari portal properti daring (Rumah123) dan broker lapangan (Product Knowledge developer).
2. **Membuktikan efektivitas Feature Engineering** melalui *Ablation Study* (perbandingan sebelum vs sesudah rekayasa fitur).
3. **Menemukan model regresi terbaik** dengan membandingkan **Random Forest** dan **XGBoost** yang dioptimasi menggunakan **Kombinasi RandomizedSearchCV & GridSearchCV (5-Fold Cross Validation)**.
4. **Mengekspor model terbaik (`model_terbaik.joblib`)** yang siap dipakai pada aplikasi web di folder terpisah.

---

## 🔄 Alur Kerja Sistem (Pipeline Data)

Alur pengerjaan dilakukan secara bertahap dan terstruktur:

```
[Sumber 1: Web Rumah123]  (3.500 listing) ──┐
                                             ├──> [01_preprocessing.py] ──> Data Cleaning (Filter Anomali)
[Sumber 2: Data Broker PK] (30 unit)      ──┘                                   │
                                                                                ├──> Feature Engineering (4 Fitur Baru)
                                                                                │
                                           ┌────────────────────────────────────┴────────────────────────────────────┐
                                           │                                                                         │
                                [dataset_tanpa_fe.csv]                                                    [dataset_dengan_fe.csv]
                                   (15 Kolom Dasar)                                                          (19 Kolom Lengkap)
                                           │                                                                         │
                                           └────────────────────────────────────┬────────────────────────────────────┘
                                                                                │
                                                                   [02_model_experiment.py]
                                                                                │
                                                            Tuning: RandomizedSearchCV + GridSearchCV (5-Fold CV)
                                                                                │
                                           ┌────────────────────────────────────┴────────────────────────────────────┐
                                           │                                                                         │
                               [Random Forest Baseline]                                                   [XGBoost Juara + FE]
                                   (R² = 0.72 - 0.73)                                                        (R² = 0.7679)
                                                                                                                     │
                                                                                                        [model_terbaik.joblib]
                                                                                                        (Siap dipakai Web App)
```

---

## 🛠️ Bedah 4 Fitur Rekayasa (Feature Engineering)

Dalam penelitian ini, dibuat 4 fitur turunan baru yang terbukti secara signifikan meningkatkan akurasi model:

| No | Nama Fitur Baru | Rumus / Cara Perhitungan | Alasan Ilmiah Pembuatan Fitur |
|---|---|---|---|
| 1 | `kecamatan` | Pencocokan nama perumahan (rule-based) & titik terdekat (GPS) | Di Surabaya, nilai tanah sangat dipengaruhi oleh zonasi wilayah (misal Surabaya Barat & Timur lebih mahal dibanding area pinggiran). Fitur ini mengelompokkan listing ke 31 kecamatan resmi. |
| 2 | `jarak_ke_pusat_kota` | Rumus Haversine ke Balai Kota Surabaya (`-7.2575, 112.7521`) | Semakin dekat properti ke pusat pemerintahan/bisnis Kota Surabaya, semakin tinggi valuasi harganya (*Urban Economics Theory*). |
| 3 | `rasio_bangunan_tanah` | $\frac{\text{Luas Bangunan}}{\text{Luas Tanah}}$ | Mengukur intensitas pemanfaatan lahan (Koefisien Dasar Bangunan). Menjelaskan apakah properti berupa rumah tumbuh/banyak halaman, atau bangunan bertingkat padat. |
| 4 | `total_ruangan` | $\text{Kamar Tidur} + \text{Kamar Mandi}$ | Menunjukkan kapasitas riil hunian untuk menampung anggota keluarga secara fungsional. |

> **⚠️ Catatan Penting Soal "Harga per Meter Persegi":**  
> Mengapa "Harga per m²" tidak boleh dimasukkan sebagai fitur prediksi?  
> Karena harga per m² dihitung dari $\frac{\text{Harga}}{\text{Luas Tanah}}$. Jika dimasukkan sebagai fitur input, model akan mengalami kebocoran data (*Data Leakage* / menyontek kunci jawaban), sehingga hasil akurasi menjadi palsu.

---

## 📊 Hasil Eksperimen Resmi (Ablation Study)

Hasil pengujian pada data uji (*Test Set* 20%) setelah melalui Hyperparameter Tuning dengan 5-Fold Cross Validation:

| Model | Skenario Fitur | R² Score | MAE (Juta Rp) | RMSE (Juta Rp) | MAPE (%) | Keterangan |
|---|---|:---:|:---:|:---:|:---:|---|
| **Random Forest** | Tanpa FE (Baseline) | 0.7338 | Rp 1.689,60 | Rp 4.936,81 | 28.78% | Baseline Model |
| **Random Forest** | DENGAN FE | 0.7246 | Rp 1.668,96 | Rp 5.021,80 | 28.22% | Evaluasi Pembanding |
| **XGBoost** | Tanpa FE (Baseline) | 0.7598 | Rp 1.645,67 | Rp 4.689,48 | 28.50% | XGBoost Standar |
| **XGBoost** | **DENGAN FE** | **0.7679** | **Rp 1.568,07** | **Rp 4.610,26** | **26.67%** | 🏆 **Model Terbaik Skripsi** |

### Kesimpulan Ilmiah:
1. **Keunggulan XGBoost**: Algoritma XGBoost terbukti mengungguli Random Forest di seluruh metrik evaluasi (R² lebih tinggi 0.0433, MAE lebih hemat Rp 100+ Juta).
2. **Pengaruh Nyata Feature Engineering**: Penambahan 4 fitur rekayasa berhasil menaikkan nilai R² XGBoost dari **0.7598 menjadi 0.7679**, serta memangkas rata-rata selisih kesalahan prediksi (MAE) sebesar **Rp 77,6 Juta per rumah**.

## 🔬 Uji Signifikansi Statistik (Repeated Cross-Validation)

Untuk menguji keabsahan bahwa peningkatan performa dari *Feature Engineering* bukan karena kebetulan (*chance*), dilakukan pengujian **Repeated K-Fold Cross Validation** (5-Fold $\times$ 10 Repeats = 50 Folds) yang dilanjutkan dengan uji beda berpasangan (**Paired t-test** pada $\alpha = 0.05$):

| Model | R² Tanpa FE (Mean ± Std) | R² Dengan FE (Mean ± Std) | t-Statistic | p-Value | Kesimpulan Statistik |
|---|:---:|:---:|:---:|:---:|---|
| **Random Forest** | 0.7585 ± 0.0369 | 0.7658 ± 0.0364 | 4.4988 | 4.2092e-05 | **Signifikan** ($p < 0.05$) |
| **XGBoost** | 0.7745 ± 0.0370 | **0.7895 ± 0.0375** | 5.5133 | 1.3031e-06 | **Signifikan** ($p < 0.05$) |

### Interpretasi Hasil:
1. **P-Value Sangat Kecil ($p < 0.0001$)**: Nilai $p$-value untuk kedua model jauh di bawah batas signifikansi 0.05. Hal ini membuktikan secara ilmiah bahwa penambahan 4 fitur rekayasa secara konsisten dan signifikan meningkatkan performa prediksi harga rumah.
2. **Stabilitas Model**: Standar deviasi yang relatif rendah (~0.037) di seluruh 50 fold membuktikan kedua model memiliki generalisasi yang stabil dan tahan terhadap variasi sampel data.
3. **XGBoost Tetap Unggul**: Model XGBoost dengan Feature Engineering menghasilkan rata-rata $R^2$ tertinggi ($0.7895$), mempertegas posisinya sebagai model terbaik dalam penelitian ini.

---

## 🎯 Top 10 Fitur Paling Menentukan Harga Rumah (Feature Importance)

Berdasarkan bobot kepentingan fitur (*Feature Importance*) dari model XGBoost terbaik (`output/feature_importance_xgboost.csv`):

| Rank | Nama Fitur | Importance (%) |
|:---:|---|:---:|
| 1 | `luas_tanah` | 16.73% |
| 2 | `kamar_mandi` | 10.54% |
| 3 | `luas_bangunan` | 8.96% |
| 4 | `total_ruangan` | 5.04% |
| 5 | `kecamatan_Pakal` | 3.90% |
| 6 | `kecamatan_Simokerto` | 3.15% |
| 7 | `kecamatan_Rungkut` | 3.08% |
| 8 | `kecamatan_Asemrowo` | 3.01% |
| 9 | `kecamatan_Mulyorejo` | 2.72% |
| 10 | `kecamatan_Tenggilis Mejoyo` | 2.24% |

---

## 📁 Struktur Folder & Berkas

```text
DATASET-RUMAH/
├── scraper.py                 # Bot pengumpul data Rumah123 (Playwright Stealth)
├── 01_preprocessing.py        # Pembersih data, pemetaan kecamatan, dan rekayasa fitur
├── 02_model_experiment.py     # Tuning dan evaluasi model RF vs XGBoost (5-Fold CV)
├── 03_cv_significance_test.py # Uji Repeated Cross-Validation & Paired t-test
├── README.md                  # Dokumentasi dan panduan lengkap
│
├── data/                      # Seluruh berkas dataset mentah & olahan
│   ├── rumah123_surabaya_raw_2026-09-14.csv  # 3.500 data scraping mentah
│   ├── data_broker_surabaya_raw.csv          # 30 data unit broker developer
│   ├── data_broker_surabaya.xlsx             # Spreadsheet data broker
│   ├── dataset_tanpa_fe.csv                  # Dataset baseline (15 kolom)
│   ├── dataset_dengan_fe.csv                 # Dataset lengkap (19 kolom)
│   └── metadata_kecamatan.json               # Koordinat & statistik 31 kecamatan
│
└── output/                    # Artefak hasil pelatihan & evaluasi model
    ├── model_terbaik.joblib                  # Model biner XGBoost juara untuk web app
    ├── tabel_hasil_eksperimen.csv            # Tabel perbandingan 4 skenario
    ├── cv_significance_results.csv           # Hasil uji Repeated CV & t-test
    ├── feature_importance_xgboost.csv        # Peringkat faktor penentu harga
    └── best_hyperparameters.json             # Parameter optimal hasil tuning
```

---

## 🚀 Cara Menjalankan Ulang (Jika Diperlukan)

Jika ingin mendemonstrasikan proses di depan dosen:

1. **Jalankan Preprocessing & Feature Engineering**:
   ```bash
   python 01_preprocessing.py
   ```
2. **Jalankan Eksperimen & Tuning Model**:
   ```bash
   python 02_model_experiment.py
   ```
3. **Jalankan Uji Signifikansi Statistik (Repeated CV & Paired t-test)**:
   ```bash
   python 03_cv_significance_test.py
   ```

Semua script sudah dirancang otomatis dan siap pakai tanpa memerlukan konfigurasi tambahan.

