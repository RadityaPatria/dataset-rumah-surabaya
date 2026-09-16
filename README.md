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
                                             ├──> [01_preprocessing.py] ──> Imputasi Median/Modus & Outlier Removal (IQR)
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
                                   (R² = 0.81 - 0.82)                                                        (R² = 0.8534)
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
| **Random Forest** | Tanpa FE (Baseline) | 0.8122 | Rp 709,80 | Rp 1.189,10 | 25.67% | Baseline Model |
| **Random Forest** | DENGAN FE | 0.8155 | Rp 693,81 | Rp 1.178,45 | 24.87% | Evaluasi Pembanding |
| **XGBoost** | Tanpa FE (Baseline) | 0.8282 | Rp 683,74 | Rp 1.137,29 | 23.76% | XGBoost Standar |
| **XGBoost** | **DENGAN FE** | **0.8534** | **Rp 655,65** | **Rp 1.050,53** | **23.16%** | 🏆 **Model Terbaik Skripsi** |

### Kesimpulan Ilmiah:
1. **Keunggulan XGBoost**: Algoritma XGBoost terbukti mengungguli Random Forest di seluruh metrik evaluasi (R² lebih tinggi hingga 0.0379, MAE lebih hemat Rp 38+ Juta).
2. **Pengaruh Nyata Feature Engineering**: Penambahan 4 fitur rekayasa berhasil menaikkan nilai R² XGBoost dari **0.8282 menjadi 0.8534**, memangkas rata-rata kesalahan prediksi (MAE) sebesar **Rp 28,09 Juta per rumah**, dan menurunkan tingkat persentase kesalahan (MAPE) ke level terendah **23.16%**.
3. **Dampak Pembersihan Outlier (IQR)**: Penghapusan 391 data pencilan ekstrem berhasil menurunkan MAE dari sebelumnya Rp 1,57 Miliar menjadi Rp 655 Juta, dan memangkas RMSE dari Rp 4,62 Miliar menjadi Rp 1,05 Miliar (penurunan error kuadratik > 77%).

## 🔬 Uji Signifikansi Statistik (Repeated Cross-Validation)

Untuk menguji keabsahan bahwa peningkatan performa dari *Feature Engineering* bukan karena kebetulan (*chance*), dilakukan pengujian **Repeated K-Fold Cross Validation** (5-Fold $\times$ 10 Repeats = 50 Folds) yang dilanjutkan dengan uji beda berpasangan (**Paired t-test** pada $\alpha = 0.05$):

| Model | R² Tanpa FE (Mean ± Std) | R² Dengan FE (Mean ± Std) | t-Statistic | p-Value | Kesimpulan Statistik |
|---|:---:|:---:|:---:|:---:|---|
| **Random Forest** | 0.8018 ± 0.0186 | 0.8067 ± 0.0183 | 8.7293 | 1.4984e-11 | **Signifikan** ($p < 0.05$) |
| **XGBoost** | 0.8109 ± 0.0194 | **0.8236 ± 0.0163** | 9.2250 | 2.7311e-12 | **Signifikan** ($p < 0.05$) |

### Interpretasi Hasil:
1. **P-Value Sangat Signifikan ($p < 10^{-11}$)**: Nilai $p$-value untuk kedua model jauh di bawah batas signifikansi 0.05 ($p = 2.73 \times 10^{-12}$ untuk XGBoost dan $p = 1.50 \times 10^{-11}$ untuk RF). Hal ini membuktikan secara ilmiah bahwa penambahan 4 fitur rekayasa secara konsisten dan signifikan meningkatkan performa prediksi harga rumah.
2. **Stabilitas Generalisasi Sangat Tinggi**: Standar deviasi yang sangat rendah (~0.016 - 0.019) di seluruh 50 fold membuktikan kedua model memiliki generalisasi yang sangat stabil dan andal terhadap variasi sampel data.
3. **XGBoost Tetap Unggul Konsisten**: Model XGBoost dengan Feature Engineering menghasilkan rata-rata $R^2$ tertinggi ($0.8236$), mempertegas posisinya sebagai model terbaik dalam penelitian ini.

---

## 🎯 Top 10 Fitur Paling Menentukan Harga Rumah (Feature Importance)

Berdasarkan bobot kepentingan fitur (*Feature Importance*) dari model XGBoost terbaik (`output/feature_importance_xgboost.csv`):

| Rank | Nama Fitur | Importance (%) |
|:---:|---|:---:|
| 1 | `luas_bangunan` | 18.00% |
| 2 | `kamar_mandi` | 9.14% |
| 3 | `luas_tanah` | 8.60% |
| 4 | `kecamatan_Rungkut` | 4.63% |
| 5 | `total_ruangan` | 4.20% |
| 6 | `kecamatan_Mulyorejo` | 3.71% |
| 7 | `kecamatan_Benowo` | 2.96% |
| 8 | `lantai` | 2.94% |
| 9 | `kecamatan_Asemrowo` | 2.93% |
| 10 | `kecamatan_Gubeng` | 2.69% |

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

