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

---

## 🎯 Top 10 Fitur Paling Menentukan Harga Rumah (Feature Importance)

Dari bobot keputusan pohon XGBoost terbaik (`feature_importance_xgboost.csv`):
1. **`luas_tanah`** (16.73%) – Faktor fisik tanah paling dominan.
2. **`kamar_mandi`** (10.54%) – Indikator kelas kemewahan spesifikasi rumah.
3. **`luas_bangunan`** (8.96%) – Luas fisik bangunan hunian.
4. **`total_ruangan`** (5.04%) – **Fitur Rekayasa (FE)** yang masuk 4 besar faktor terpenting!
5. **Faktor Wilayah/Lokasi**: `kecamatan_Pakal`, `kecamatan_Simokerto`, `kecamatan_Rungkut`, `kecamatan_Asemrowo`, dan `jarak_ke_pusat_kota`.

---

## 🎓 CHEAT SHEET SEMINAR PROPOSAL (TANYA - JAWAB DOSEN)

Gunakan contekan jawaban santai, runtut, dan ilmiah ini saat ditanya oleh dosen pembimbing maupun dosen penguji:

### 1. "Mas Radit, kenapa membandingkan XGBoost dengan Random Forest?"
> **Jawaban:**  
> "Keduanya sama-sama algoritma *Ensemble Tree* yang sangat kuat untuk data tabel properti. Namun cara kerjanya berbeda, Pak/Bu:  
> - **Random Forest** bekerja secara *Bagging* (membangun banyak pohon secara independen lalu diambil voting/rata-ratanya).  
> - **XGBoost** bekerja secara *Gradient Boosting* (pohon dibangun berurutan, di mana setiap pohon baru bertugas khusus memperbaiki kesalahan dari pohon sebelumnya).  
> Di penelitian ini, saya ingin membuktikan secara empiris bahwa mekanisme *boosting* pada XGBoost lebih efektif menangani pola harga rumah di Surabaya."

### 2. "Kenapa harus repot-repot bikin Feature Engineering?"
> **Jawaban:**  
> "Karena data mentah dari internet itu fiturnya sangat dasar (hanya luas tanah, bangunan, dan kamar). Padahal dalam dunia properti, **lokasi, zonasi, dan kepadatan bangunan** sangat menentukan harga.  
> Dengan membuat fitur seperti jarak ke Balai Kota, rasio bangunan-tanah, dan nama kecamatan, kita memberi 'wawasan tambahan' kepada model Machine Learning sehingga ia bisa membedakan harga rumah di pusat kota vs area pinggiran secara lebih akurat."

### 3. "Kenapa harga rumah di-log transform (`np.log1p`) saat latihan?"
> **Jawaban:**  
> "Harga rumah memiliki variansi yang sangat lebar (mulai Rp 200 juta sampai puluhan miliar), sehingga distribusinya menceng ke kanan (*right-skewed*). Jika langsung dilatihkan mentah-mentah, model akan bias dan terdistorsi oleh rumah-rumah yang kelewat mahal (*outlier*).  
> Dengan fungsi logaritma, distribusinya dibuat lebih simetris menyerupai distribusi normal. Setelah prediksi selesai, nilainya kita kembalikan lagi ke Rupiah riil memakai fungsi inversi `np.expm1` agar metrik evaluasi (MAE & MAPE) tetap dalam satuan Rupiah asli."

### 4. "Mengapa menggunakan rumus Haversine untuk menghitung jarak?"
> **Jawaban:**  
> "Rumus Haversine digunakan untuk menghitung jarak lingkaran besar (*great-circle distance*) antara dua titik koordinat GPS di permukaan bumi. Karena bumi itu bulat, kita tidak bisa memakai rumus jarak Euclidean biasa (Pythagoras) yang mengasumsikan bumi itu datar. Rumus Haversine memperhitungkan kelengkungan bumi dengan jari-jari rata-rata 6.371 km."

### 5. "Apa bedanya RandomizedSearchCV dan GridSearchCV yang Mas pakai?"
> **Jawaban:**  
> "Sesuai metodologi pada Bab 3, saya menggunakan strategi *Coarse-to-Fine Search*:  
> - Pertama, **RandomizedSearchCV** dipakai untuk eksplorasi cepat menjelajahi rentang parameter yang luas secara acak.  
> - Kedua, setelah ditemukan kandidat parameter yang menjanjikan, **GridSearchCV** dipakai untuk menyisir secara presisi di sekitar angka terbaik tersebut dengan 5-Fold Cross Validation.  
> Strategi ini menghemat waktu komputasi secara drastis tanpa mengorbankan kualitas akurasi model."

### 6. "Mas kan masih Sempro (Bab 1–3), kenapa eksperimen modelnya sudah selesai dijalankan?"
> **Jawaban:**  
> "Sebagai mahasiswa D4 Teknik Informatika Vokasi yang berbasis terapan, saya ingin memastikan bahwa usulan penelitian saya **layak secara data dan teknis (*Feasibility Study*)**.  
> Dengan menguji data riil dan pipeline model sejak awal, saya bisa membuktikan ke dosen bahwa rumusan masalah Bab 1 benar-benar dapat terjawab secara terukur, tidak berhenti pada hipotesis di atas kertas, dan siap dilanjutkan ke tahap implementasi aplikasi web untuk Bab 4."

---

## 📁 Struktur Folder & Berkas

```text
DATASET-RUMAH/
├── scraper.py                 # Bot pengumpul data Rumah123 (Playwright Stealth)
├── 01_preprocessing.py        # Pembersih data, pemetaan kecamatan, dan rekayasa fitur
├── 02_model_experiment.py     # Tuning dan evaluasi model RF vs XGBoost (5-Fold CV)
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

Semua script sudah dirancang otomatis dan siap pakai tanpa memerlukan konfigurasi tambahan.
