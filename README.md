# Panduan penelitian harga rumah Surabaya

Judul: **Prediksi Harga Rumah di Kota Surabaya Menggunakan Algoritma XGBoost Berbasis Feature Engineering**.

README ini menjelaskan implementasi aktual untuk memahami proyek dan menyusun Bab III. XGBoost Regressor menjadi model utama; Random Forest menjadi pembanding. Penelitian membandingkan fitur lokasi dan fisik sebelum dan sesudah feature engineering (FE) pada baris data dan pembagian train-test yang sama.

**Keputusan dataset terbaru:** `alamat_teks` hanya disimpan pada data mentah dan digunakan selama cleaning/FE. Kolom tersebut sudah dihapus dari kedua CSV hasil preprocessing. GPS dipertahankan pada dataset tanpa FE, tetapi dihapus pada dataset FE dan digantikan oleh `kecamatan` serta `jarak_ke_pusat_kota`. FE juga menambah `rasio_bangunan_tanah` dan `luas_bangunan_per_lantai`. Kondisi properti dan fasilitas tidak terpilih pada revisi ini; kelurahan tidak digunakan.

## 1. Isi proyek dan fungsi setiap file

```text
DATASET-RUMAH/
|-- README.md                 Panduan alur dan metode penelitian
|-- requirements.txt          Versi pustaka Python
|-- scraper.py                Pengambilan listing Rumah123
|-- pipeline.py               Cleaning, fitur lokasi, training, evaluasi
|-- data/
|   |-- rumah123_surabaya_raw_2026-09-21_balanced.csv
|   |-- data_broker_surabaya.xlsx
|   |-- dataset_tanpa_fe.csv
|   `-- dataset_dengan_fe.csv
`-- hasil/
    |-- hasil_penelitian.xlsx
    `-- model_xgboost_fe.joblib
```

| File | Isi dan tujuan |
|---|---|
| CSV `raw` | 4.000 listing hasil ekstraksi Rumah123, sebelum cleaning penelitian; sumber untuk mengulang preprocessing |
| `data_broker_surabaya.xlsx` | 30 unit data broker sebagai sumber tambahan; dibaca satu kali agar tidak terhitung ganda |
| `dataset_tanpa_fe.csv` | 3.728 baris, 12 kolom; baseline dengan koordinat |
| `dataset_dengan_fe.csv` | 3.728 baris, 14 kolom; pasangan baseline dengan lokasi dan dua fitur fisik turunan |
| `hasil_penelitian.xlsx` | Lima sheet: Model, Parameter, Data, Importance, Riwayat |
| `model_xgboost_fe.joblib` | Pipeline terlatih: imputasi, encoding, dan XGBoost FE untuk prediksi |

Folder tersembunyi `.git` menyimpan riwayat versi. `__pycache__` dapat muncul otomatis ketika Python dijalankan; bukan data atau hasil penelitian.

## 2. Gambaran alur dari awal sampai akhir

```text
Rumah123 melalui scraper.py + Excel broker
                  |
                  v
Gabung sumber -> deduplikasi -> filter lokasi/lelang/jenis properti
                  |
                  v
Normalisasi nilai -> batas minimum harga/luas -> outlier IQR log
                  |
                  v
Audit kondisi + kecamatan/jarak + rasio fisik tanpa memakai harga
                  |
Seleksi atribut pada 80% training, validasi 5-fold
          +-------+-------+
          |               |
          v               v
CSV tanpa FE          CSV dengan FE
GPS tersedia          Lokasi + rasio fisik; GPS dihapus
Alamat dihapus        Alamat dihapus
          |               |
          +-------+-------+
                  |
                  v
Split indeks yang sama: training 80%, testing 20%
                  |
                  v
Gunakan atribut yang sudah dipilih saat preprocessing
                  |
                  v
Empat skenario RF/XGBoost dengan/tanpa FE
Imputasi + encoding di dalam setiap fold training
RandomizedSearchCV -> GridSearchCV -> refit pada seluruh training
                  |
                  v
Prediksi testing -> kembalikan harga log ke rupiah -> hitung metrik
                  |
                  v
Simpan Excel evaluasi + pipeline XGBoost FE
```

Dua CSV bersih memuat seluruh baris, tetapi seleksi kolomnya hanya memakai indeks 80% training. Sel kosong belum diimputasi secara global. Keduanya merupakan dua representasi dari observasi yang sama, bukan dua kelompok rumah berbeda. Pembentukan fitur lokasi memakai aturan tetap dan tidak memakai harga target.

## 3. Pengumpulan data

### 3.1 Sumber dan unit observasi

Unit observasi adalah **listing penawaran rumah**, bukan transaksi jual beli yang sudah selesai. `harga` adalah harga penawaran dari sumber. Data mentah saat ini terdiri atas 4.000 listing Rumah123 dalam file bertanggal 21 September 2026 dan 30 unit broker. Tanggal pengumpulan broker harus mengacu pada catatan sumber; jangan mengarang tanggalnya dari nama file Rumah123.

CSV mentah adalah hasil ekstraksi atribut terstruktur, bukan salinan HTML utuh. Scraper telah memeriksa URL, harga positif, serta ketersediaan luas sebelum menyimpan baris. Jadi istilah "mentah" di sini berarti sebelum cleaning penelitian, bukan tanpa validasi sama sekali.

### 3.2 Cara kerja scraper

`python scraper.py --target 4000` menggunakan Playwright/Chromium untuk membuka halaman penjualan rumah di Surabaya dan membaca payload listing dari halaman.

1. Pencarian dibagi ke lima wilayah: Timur, Barat, Utara, Selatan, dan Pusat, dengan daftar 31 kecamatan pada `REGIONS` di kode.
2. Target 4.000 dibagi menjadi 800 URL unik per wilayah pencarian. Halaman dicari bergiliran antarwilayah dan antarkecamatan (round-robin).
3. Atribut listing diekstrak: judul, harga, luas, kamar, lantai, carport, furnished, fasilitas, alamat, koordinat, dan URL.
4. URL yang sudah dikumpulkan tidak disimpan ulang. Baris ditambahkan ke satu CSV bertanggal.
5. Jika file tanggal yang sama sudah ada, scraper membaca URL dan kuota yang terkumpul untuk melanjutkan proses.
6. Jeda standar 3,5-6,5 detik; batas 250 halaman per kecamatan. Pencarian berhenti ketika kuota tercapai, pencarian habis, atau situs membatasi akses melalui HTTP 403/429 maupun halaman challenge.

Ini **pengambilan sampel berbasis kuota pencarian**, bukan sampel acak seluruh rumah Surabaya. Kuota seimbang tidak berarti distribusi kecamatan atau populasi pasar seimbang. `wilayah_scraping` menandai kelompok pencarian, bukan validasi batas wilayah rumah. Setelah cleaning, jumlah tiap wilayah tidak lagi harus 800.

### 3.3 Struktur data mentah

CSV Rumah123 lama mempunyai 17 kolom: judul_listing, harga, luas_tanah, luas_bangunan, kamar_tidur, kamar_mandi, lantai, carport, furnished, keamanan, taman, alamat_teks, latitude, longitude, url_listing, sumber_data, dan wilayah_scraping. Broker mempunyai 16 kolom tanpa wilayah_scraping; pipeline menambah label audit "Broker (bukan scraping)" saat penggabungan. Data mentah tidak dipotong mengikuti seleksi fitur hasil akhir.

`keamanan` dan `taman` berasal dari deteksi kata kunci pada teks/atribut listing. Nilai 0 berarti kata kunci tidak terdeteksi, bukan jaminan fasilitas tidak ada. Scraper terbaru mempertahankan furnished yang tidak tersedia sebagai kosong; data lama dapat memiliki nilai default Unfurnished sehingga ketepatannya tetap bergantung pada ekstraksi sumber.

## 4. Preprocessing: urutan, aturan, dan jumlah data

`run_preprocessing()` memilih CSV `rumah123_surabaya_raw_*.csv` terbaru menurut nama file, lalu membaca satu sumber broker. Urutan filter memengaruhi jumlah penghapusan per tahap.

| Tahap | Dihapus pada tahap ini | Sisa baris |
|---|---:|---:|
| Gabungkan Rumah123 dan broker | - | 4.030 |
| Deduplikasi | 0 | 4.030 |
| Filter indikasi luar Surabaya | 1 | 4.029 |
| Filter lelang/sitaan | 171 | 3.858 |
| Filter jenis properti di luar cakupan | 74 | 3.784 |
| Normalisasi nilai invalid menjadi kosong | 0 baris | 3.784 |
| Batas minimum harga dan luas | 8 | 3.776 |
| Gabungan outlier IQR log | 48 | 3.728 |
| FE dan ekspor dua CSV | 0 | 3.728 |

Seluruh 30 data broker tetap ada. Data bersih mencakup harga Rp200 juta-Rp65 miliar, luas tanah 20-2.500 m2, dan luas bangunan 24-1.600 m2. Jumlah dan rentang ini adalah hasil dataset saat ini, bukan target yang dijamin untuk scraping berikutnya.

### 4.1 Deduplikasi dan pembatasan cakupan

- URL tidak kosong: gunakan URL sebagai identitas listing.
- URL kosong: gunakan gabungan judul, alamat, harga, luas tanah/bangunan, kamar tidur/mandi, dan sumber data. Alamat masih tersedia pada tahap internal ini.
- Lokasi luar kota: periksa indikasi Mojokerto, Sidoarjo, Gresik, Lamongan, Bangkalan, Mojosari, dan Krian pada judul/alamat. Penyebutan tetangga pada judul dapat dipertahankan jika alamat secara eksplisit menunjukkan kecamatan Surabaya dan alamat tidak menyebut daerah luar tersebut.
- Lelang: periksa kata lelang, sita/sitaan, disita, penyitaan, atau eksekusi pada judul.
- Jenis properti: pola judul menyaring indikasi rumah kos, kos aktif/eksklusif, homestay, bekas penginapan, rumah gudang, ruko, hotel, apartemen, dan gedung. Bagian setelah kata seperti dekat, cocok, untuk, atau potensi tidak dijadikan bukti jenis properti.

Contoh "Rumah Kost Aktif" disaring, sedangkan "Rumah Cocok Untuk Kos" tidak disaring hanya karena kata kos. Aturan ini berbasis teks dan belum menjamin seluruh iklan salah kategori terdeteksi. Daftar judul yang disaring tersimpan di sheet Data.

### 4.2 Normalisasi dan missing value

`normalize_attributes()` mengubah numerik yang tidak dapat dibaca atau bernilai infinity menjadi kosong. Aturan operasional berikut mengosongkan nilai atribut, bukan langsung menghapus rumah:

| Atribut | Aturan |
|---|---|
| Kamar tidur/mandi, lantai, carport | Nilai negatif atau jumlah pecahan menjadi kosong |
| Lantai | Di luar 1-6 menjadi kosong untuk verifikasi |
| Carport | `carport * 10 > luas_tanah` menjadi kosong; asumsi deteksi minimum 10 m2/mobil |
| Kamar tidur | `kamar_tidur * 6 > luas_bangunan` menjadi kosong |
| Kamar mandi | `kamar_mandi * 2 > luas_bangunan` menjadi kosong |
| Keamanan, taman | Hanya 0 dan 1 yang diterima |
| GPS | Pasangan harus berada dalam latitude [-7,5; -7,1] dan longitude [112,5; 112,9]; jika gagal, kedua nilai dikosongkan |
| Furnished | Rapikan kapital/spasi; pertahankan Unfurnished, Semi Furnished, Furnished; Fully Furnished diseragamkan menjadi Furnished; nilai lain kosong |

Batas tersebut adalah asumsi operasional penelitian, bukan standar universal rumah. Kotak rentang GPS bukan pemeriksaan batas administratif Surabaya. Pada tahap normalisasi saat ini, 353 pasangan koordinat terisi tidak valid; sebagian longitude justru berisi angka latitude. Nilai seperti "Bagus" atau "butuh renovasi" tidak dianggap kategori furnished.

**CSV tetap boleh memiliki sel kosong.** Median numerik dan modus kategori baru dihitung di dalam pipeline training. Setiap fold CV mempelajari imputasi dari subset training fold tersebut, kemudian menerapkannya pada validation fold. Data testing tidak menentukan median/modus. Harga target tidak pernah diimputasi.

### 4.3 Batas minimum dan outlier

Baris harus memiliki harga minimal Rp100 juta, luas tanah minimal 15 m2, serta luas bangunan minimal 15 m2. Nilai kosong pada ketiganya gagal pemeriksaan ini.

Untuk masing-masing kolom `harga`, `luas_tanah`, dan `luas_bangunan`:

```text
z = ln(1 + nilai)
IQR = Q3(z) - Q1(z)
batas bawah = Q1(z) - 1,5 * IQR
batas atas  = Q3(z) + 1,5 * IQR
```

Baris dihapus jika salah satu dari tiga atribut berada di luar batasnya. Mask digabung, sehingga satu baris yang bermasalah pada beberapa atribut hanya dihitung sekali. Ini bukan log dari rasio harga terhadap luas. Nilai yang diekspor tetap rupiah/m2, bukan log.

**Batas metodologis:** IQR dipelajari dari seluruh populasi sebelum train-test split, termasuk harga. Karena itu evaluasi bersifat kondisional pada populasi tersaring dan belum sepenuhnya bebas pengaruh preprocessing global. Jangan menulis bahwa seluruh preprocessing sudah dipelajari hanya dari training; klaim tersebut berlaku untuk imputasi dan encoding, bukan filter IQR ini.

## 5. Isi kedua dataset hasil preprocessing

### 5.1 Atribut bersama yang dipertahankan

| Kolom | Makna/satuan | Peran |
|---|---|---|
| `judul_listing` | Judul iklan | Metadata untuk pemeriksaan, tidak masuk model |
| `harga` | Harga penawaran, rupiah | Target prediksi |
| `luas_tanah` | Luas tanah, m2 | Fitur numerik |
| `luas_bangunan` | Luas bangunan, m2 | Fitur numerik |
| `kamar_tidur` | Jumlah kamar tidur | Fitur numerik |
| `kamar_mandi` | Jumlah kamar mandi | Fitur numerik |
| `lantai` | Jumlah lantai | Fitur numerik |
| `carport` | Kapasitas parkir dari atribut sumber | Fitur numerik; tetap dipilih oleh validasi |
| `furnished` | Unfurnished / Semi Furnished / Furnished | Fitur kategorikal dengan cakupan tinggi |
| `sumber_data` | Asal listing | Metadata, tidak masuk model |

### 5.2 Kolom khusus setiap skenario

| Dataset | Kolom tambahan | Jumlah total kolom CSV |
|---|---|---:|
| Tanpa FE | latitude, longitude | 12 |
| Dengan FE | kecamatan, jarak_ke_pusat_kota, rasio_bangunan_tanah, luas_bangunan_per_lantai | 14 |

Baseline memakai 9 atribut input sebelum encoding, sedangkan FE memakai 11. Tiga kolom lain merupakan target dan dua metadata. Kedua CSV tetap memiliki 3.728 baris rumah yang sama dalam urutan identik. GPS tidak ada pada CSV FE; alamat tidak ada pada kedua CSV bersih.

**Kondisi properti, keamanan, dan taman tidak masuk CSV/model hasil revisi ini.** Kondisi hanya tersedia untuk sekitar 11% data training sehingga gagal kebijakan kelengkapan minimum 50%. Keamanan dan taman tidak dipilih dalam perbandingan kandidat training. Data sumber asli serta ringkasan audit tetap disimpan; nilai kosong tidak diubah menjadi klaim kondisi baik.

Carport memiliki 1.392 nilai kosong (sekitar 37,34%), tetapi kandidat yang mempertahankannya lebih baik pada validasi. Karena itu atribut tidak dibuang hanya berdasarkan jumlah kosong. Furnished hanya mempunyai 8 nilai kosong setelah normalisasi dan tetap dipakai; variabel perabot ini tidak boleh diklaim sebagai kondisi fisik bangunan.

Ukuran schema di atas berlaku untuk seleksi pada sumber sekarang. Bila sumber diperbarui, `select_reliable_features()` menguji ulang kandidat pada indeks training, dan jumlah kolom dapat mengikuti hasil seleksi. `run_experiments()` membaca schema terpilih dari CSV agar kolom yang sudah dibuang tidak masuk kembali.

## 6. Feature engineering lokasi dan fisik

### 6.1 Kecamatan

`map_text_to_kecamatan()` membaca bagian alamat yang berupa nama kecamatan administratif, dipisahkan koma/titik koma, atau label eksplisit Kec./Kecamatan. Alias Gununganyar diseragamkan ke Gunung Anyar dan Pabean Cantikan ke Pabean Cantian. Label Kenjeran tetap Kenjeran, bukan Bulak.

Jika label tidak ditemukan, `get_nearest_kecamatan()` memilih centroid kecamatan dengan jarak terdekat dari GPS valid. Ini perkiraan, bukan point-in-polygon dengan peta batas resmi. Jika keduanya tidak tersedia, kategori menjadi "Tidak diketahui"; tidak ada kecamatan default yang dipaksakan.

Pada dataset bersih sekarang: 3.720 kecamatan berasal dari label alamat dan 8 dari perkiraan GPS. Kelurahan tidak ditambahkan karena data belum konsisten menyediakannya. Nama jalan/kelurahan juga tidak dibuat-buat dari label kecamatan.

### 6.2 Jarak ke pusat kota

Titik acuan tetap di kode adalah latitude -7,2575 dan longitude 112,7521. Jarak dihitung memakai Haversine dengan radius bumi 6.371 km:

```text
a = sin^2((phi2 - phi1)/2)
    + cos(phi1) * cos(phi2) * sin^2((lambda2 - lambda1)/2)
d = 2 * 6371 * atan2(sqrt(a), sqrt(1-a))
```

Phi adalah latitude dan lambda longitude dalam radian. Hasil dibulatkan dua desimal kilometer. Ini jarak permukaan bumi langsung, bukan rute jalan, waktu tempuh, atau akses ke banyak fasilitas.

Urutan sumber titik: GPS valid listing, kemudian centroid kecamatan jika GPS tidak tersedia. Jika keduanya tidak tersedia, jarak dibiarkan kosong untuk imputasi training. Centroid tersedia sebagai konstanta dalam kode, bukan hasil geocoding alamat lengkap. Pin listing bisa mewakili kawasan sehingga ketelitian jarak terbatas.

**Setelah FE selesai, latitude dan longitude tidak diekspor ke CSV FE dan tidak masuk model FE.** CSV baseline masih mempertahankan keduanya. Perbandingan ini menguji penggantian representasi GPS dengan kecamatan+jarak, bukan penambahan FE sambil mempertahankan seluruh fitur GPS.

### 6.3 Dua fitur fisik dari atribut yang terukur

| Fitur | Rumus | Arti dan batas interpretasi |
|---|---|---|
| `rasio_bangunan_tanah` | luas_bangunan / luas_tanah | Perbandingan luas bangunan total terhadap luas tanah, tanpa satuan |
| `luas_bangunan_per_lantai` | luas_bangunan / lantai | Rata-rata luas bangunan per lantai, m2/lantai |

Contoh LT 100 m2, LB 180 m2, 2 lantai menghasilkan rasio 1,8 dan rata-rata luas 90 m2/lantai. Rasio dapat lebih dari 1 karena luas bangunan mencakup beberapa lantai. Rasio ini bukan koefisien tapak bangunan yang diukur di lapangan. Luas per lantai juga bukan ukuran pasti luas lantai dasar karena ukuran antarlantai dapat berbeda.

Rasio bangunan/tanah tersedia pada seluruh 3.728 baris setelah filter luas. Luas bangunan per lantai tersedia pada 3.602 baris (sekitar 96,62%); 126 baris tetap kosong karena jumlah lantai tidak diketahui. Pembagian hanya dilakukan untuk penyebut positif. Nilai kosong diimputasi median pada training fold, bukan ditebak sebelum split.

Kedua fitur tidak menggunakan harga, sehingga dapat dihitung untuk rumah baru sebelum memprediksi. Tidak ada harga per m2 sebagai input karena itu akan memakai target yang seharusnya belum diketahui. Hipotesisnya: rasio membantu model membaca hubungan antarukuran fisik; manfaat empirisnya diuji sebagai satu kelompok melalui CV training, bukan diasumsikan selalu meningkat.

### 6.4 Status kondisi properti dan keterbatasan variabel kualitatif

Ekstraksi kondisi dari klaim eksplisit judul tetap dilakukan untuk audit, tetapi **tidak menjadi fitur aktif pada hasil revisi**. Distribusi seluruh data bersih: Baru 276, Terawat 93, Direnovasi 36, Perlu renovasi 13, Sedang renovasi 1, dan Tidak diketahui 3.309. Hanya 419 listing yang memiliki bukti; seluruh 30 broker saat ini tidak memiliki keterangan kondisi eksplisit.

`extract_property_condition()` mengutamakan kolom kondisi eksplisit pada sumber jika kelak tersedia; jika tidak, membaca frasa judul. Negasi atau klaim bertentangan tidak dipaksakan menjadi label. Judul dan contoh bukti tetap tercatat pada sheet Data agar alasan tidak menggunakan atribut tersebut dapat dijelaskan. Tidak diketahui tidak sama dengan rusak maupun baik.

Kebijakan penelitian revisi mensyaratkan sedikitnya 50% kondisi diketahui pada **training** untuk mempertimbangkan atribut tersebut sebagai fitur utama. Cakupan aktual training sekitar 11,00%. Ambang 50% adalah keputusan operasional, bukan batas baku statistik atau bukti bahwa suatu fitur pasti tidak bermanfaat. Kandidat versi lama tetap diuji sebagai pembanding audit, tetapi tidak layak dipilih pada sumber saat ini.

Usulan jurnal tentang kondisi fisik dan riwayat renovasi **belum sepenuhnya terjawab**. Menambah label dengan menebak dari harga, umur, atau ketiadaan kata renovasi akan menghasilkan data yang tidak dapat dipertanggungjawabkan. Untuk memasukkan kembali kondisi secara kuat, perlu pengumpulan keterangan eksplisit dari detail listing atau broker dengan cakupan dan verifikasi memadai. Fitur rasio fisik merupakan perbaikan berdasarkan data yang tersedia, bukan pengganti konseptual kondisi bangunan.

## 7. Pembagian data, pemilihan fitur, dan training

### 7.1 Target dan pembagian data

Pipeline memeriksa kesamaan semua kolom bersama dan sumber data pada kedua CSV, termasuk urutan baris. Target training ditransformasikan menjadi `y_log = ln(1 + harga)` untuk memodelkan harga pada skala log. Target log tidak menggantikan harga rupiah yang disimpan di CSV.

`train_test_split(test_size=0.20, random_state=42)` menghasilkan **2.982 baris training dan 746 baris testing**. Kedua dataset serta kedua algoritma memakai indeks yang sama. Split bersifat acak, bukan time series dan bukan stratifikasi. Training memakai `KFold(n_splits=5, shuffle=True, random_state=42)` untuk validasi.

### 7.2 Seleksi atribut sebelum mengekspor CSV

`select_reliable_features()` berjalan pada tahap preprocessing setelah cleaning dan pembentukan kandidat FE. Fungsi ini membuat split 80:20 dengan indeks/seed yang sama seperti pelatihan akhir. Hanya 2.982 baris training yang dipakai untuk membandingkan kandidat; 746 baris testing tidak dinilai. Imputasi dan encoding dipelajari ulang di setiap fold kandidat.

Estimator penguji tetap: XGBoost, 200 pohon, depth 6, learning rate 0,05, seed 42. Kandidat dibandingkan berdasarkan rata-rata MSE harga log dalam 5-fold. Semakin kecil semakin baik. Kandidat terbatas berikut bukan pencarian seluruh kombinasi atribut:

| Kandidat baseline (fitur inti + GPS selalu ada) | MSE log training CV |
|---|---:|
| Tanpa carport, fasilitas, dan kondisi | 0,143826 |
| Carport, tanpa fasilitas dan kondisi | **0,140927** |
| Fasilitas, tanpa carport dan kondisi | 0,144900 |
| Carport + fasilitas, tanpa kondisi | 0,142295 |
| Carport + fasilitas + kondisi (versi lama) | 0,141545; gagal ambang cakupan kondisi |

Fasilitas berarti keamanan dan taman sebagai satu kelompok. Kandidat layak dengan rata-rata error terendah dipilih. Hasilnya mempertahankan carport dan membuang fasilitas serta kondisi. Seluruh fitur inti (luas tanah/bangunan, kamar tidur/mandi, lantai, furnished) dipertahankan; tidak dilakukan uji penghapusan setiap fitur inti.

Selanjutnya, dengan atribut bersama terpilih, bandingkan FE lokasi saja versus FE lokasi + dua rasio fisik:

| Kandidat FE | MSE log training CV |
|---|---:|
| Kecamatan + jarak | 0,134038 |
| Kecamatan + jarak + dua fitur fisik | **0,132405** |

Kelompok fitur fisik dipilih karena rata-rata error lebih rendah. Mean dan simpangan baku antarfold dicatat di sheet Data pada `seleksi_atribut_training`. Selisih ini belum merupakan bukti signifikansi statistik. Pilihan bersama diterapkan pada kedua algoritma agar skenario RF dan XGBoost memakai definisi fitur yang sama.

Pemilihan fitur memakai CV yang kemudian juga digunakan untuk tuning, bukan nested CV. Holdout tidak dipakai memilih kandidat. Namun, karena holdout telah dilihat pada pengembangan versi sebelumnya, keseluruhan proses pengembangan belum setara validasi eksternal independen.

### 7.3 Pipeline dalam setiap fold

1. Atribut numerik terpilih: `SimpleImputer(strategy="median")`.
2. Atribut kategori: `SimpleImputer(strategy="most_frequent")`, dilanjutkan `OneHotEncoder(handle_unknown="ignore")`.
3. Estimator: RandomForestRegressor atau XGBRegressor, seed 42.

Pada revisi saat ini, baseline memiliki kategori furnished; FE memiliki furnished dan kecamatan. Kondisi properti tidak masuk model. Kategori baru saat prediksi diabaikan encoder pada blok kategori yang bersangkutan. Tidak ada StandardScaler/MinMaxScaler, imputasi harga, SMOTE, PCA, atau imputasi global pada CSV; rasio fisik yang digunakan dijelaskan pada bagian 6.3.

### 7.4 Empat skenario

| Skenario | Algoritma | Representasi lokasi |
|---|---|---|
| 1 | Random Forest | Latitude + longitude |
| 2 | Random Forest | Kecamatan + jarak + dua fitur fisik |
| 3 | XGBoost | Latitude + longitude |
| 4 | XGBoost | Kecamatan + jarak + dua fitur fisik |

Atribut dasar terpilih dan indeks split sama; FE mengganti koordinat dengan lokasi terstruktur serta menambah dua fitur fisik. Masing-masing skenario melakukan tuning tersendiri; hyperparameter akhirnya dapat berbeda. Hasil akhir mengukur paket FE yang dipilih, bukan pengaruh kecamatan, jarak, atau satu rasio secara terpisah.

### 7.5 Pencarian hyperparameter

Tahap pertama: `RandomizedSearchCV`, 16 kandidat, 5-fold, random_state 42. Ruang pencariannya:

| Algoritma | Parameter | Kandidat |
|---|---|---|
| RF | n_estimators | 100, 150, 200, 250 |
| RF | max_depth | 10, 15, 20, None |
| RF | min_samples_split | 2, 5, 10 |
| RF | min_samples_leaf | 1, 2, 4 |
| XGBoost | n_estimators | 150, 200, 250, 300 |
| XGBoost | max_depth | 4, 6, 8 |
| XGBoost | learning_rate | 0,02; 0,05; 0,10 |
| XGBoost | subsample | 0,80; 0,90; 1,00 |
| XGBoost | colsample_bytree | 0,80; 1,00 |

Tahap kedua: `GridSearchCV` pada semua kombinasi nilai di sekitar hasil RandomizedSearchCV. Untuk nilai terbaik v, kandidat awal adalah v-step, v, v+step:

| Parameter | Step | Pembatasan |
|---|---:|---|
| n_estimators | 25 | Minimal 50 |
| max_depth | 1 | Minimal 1; jika terbaik None, hanya None |
| min_samples_split | 1 | Minimal 2 |
| min_samples_leaf | 1 | Minimal 1 |
| learning_rate | 0,01 | Minimal 0,01; pembulatan 3 desimal |
| subsample | 0,05 | Rentang 0,50-1,00; pembulatan 2 desimal |
| colsample_bytree | 0,05 | Rentang 0,50-1,00; pembulatan 2 desimal |

Nilai dibatasi, lalu duplikat kandidat dihapus. Karena itu jumlah kombinasi grid dapat berbeda antarskenario. Kedua tahap memakai skor **negative MSE pada target log**, bukan R2 testing atau RMSE rupiah. Pencarian berjalan paralel (`n_jobs=-1`); estimator memakai `n_jobs=1`. Parameter lain mengikuti default versi pustaka pada requirements.txt; tidak ada early stopping yang dikonfigurasi.

Kandidat terbaik GridSearchCV di-fit ulang pada seluruh 2.982 baris training. Testing tidak digunakan dalam pemilihan kandidat. Skor CV pemilihan fitur dan tuning bukan nested CV; jangan menyebutnya estimasi nested cross-validation yang tidak bias.

## 8. Evaluasi dan hasil tersimpan

Prediksi model masih berupa log harga, sehingga `harga_prediksi = exp(prediksi_log) - 1`. Metrik berikut dihitung pada 746 baris testing dalam rupiah asli:

```text
MAE  = mean(abs(y - y_pred))
RMSE = sqrt(mean((y - y_pred)^2))
R2   = 1 - sum((y - y_pred)^2) / sum((y - mean(y))^2)
MAPE = mean(abs((y - y_pred) / y)) * 100%
```

MAE, RMSE, dan R2 adalah metrik utama; MAPE tambahan. MAE/RMSE diekspor dalam **juta rupiah** dengan dua desimal, MAPE dalam persen dengan dua desimal, dan R2 empat desimal. R2 lebih tinggi lebih baik; nilai error lebih rendah lebih baik. R2 0,82 bukan berarti 82% prediksi pasti benar. RMSE lebih sensitif terhadap kesalahan besar.

Hasil terbaru setelah seleksi atribut dan penambahan FE fisik (22 September 2026):

| Model | Skenario | R2 | MAE (juta Rp) | RMSE (juta Rp) | MAPE (%) |
|---|---|---:|---:|---:|---:|
| Random Forest | Tanpa FE | 0,8209 | 1.495,14 | 3.469,81 | 28,32 |
| Random Forest | Dengan FE | 0,8216 | 1.405,77 | 3.462,93 | 27,02 |
| XGBoost | Tanpa FE | 0,8133 | 1.521,36 | 3.542,00 | 28,82 |
| XGBoost | Dengan FE | 0,8330 | 1.372,54 | 3.350,63 | 27,00 |

Pada eksekusi ini, XGBoost FE terbaik pada keempat metrik. Paket FE memperbaiki hasil kedua algoritma dibanding baseline masing-masing. Angka ini merupakan hasil satu split pengujian pada populasi tersaring, bukan bukti bahwa FE pasti lebih unggul pada seluruh data baru.

Dibanding versi sebelumnya yang memakai kondisi properti, XGBoost FE berubah dari R2 0,8300 menjadi 0,8330; MAE turun dari 1.446,03 menjadi 1.372,54 juta rupiah; RMSE dari 3.380,02 menjadi 3.350,63 juta; MAPE dari 28,00% menjadi 27,00%. Baris dan indeks split sama. Namun, atribut dan parameter terbaik berubah bersama, sehingga perbandingan ini tidak mengisolasi pengaruh penghapusan kondisi atau satu fitur tertentu. Tidak dilakukan uji signifikansi statistik.

Buka [hasil_penelitian.xlsx](hasil/hasil_penelitian.xlsx):

| Sheet | Kegunaan |
|---|---|
| Model | Tabel metrik empat skenario, hasil utama yang perlu dibaca |
| Parameter | Hyperparameter terpilih, skor CV, daftar fitur numerik/kategori yang dipakai |
| Data | Cleaning, missing value, lokasi, retensi broker, audit kondisi, dan hasil CV seleksi atribut/FE |
| Importance | Importance bawaan XGBoost FE setelah encoding; bukan SHAP atau bukti sebab-akibat |
| Riwayat | Metrik versi sebelumnya; populasi dan preprocessing bisa berbeda |

Hanya pipeline XGBoost FE yang disimpan sebagai `.joblib`, bukan empat file model. Pipeline tersimpan di-fit pada training 80%, belum di-fit ulang ke seluruh dataset. Pembagian ini mempertahankan kesesuaian model tersimpan dengan evaluasi holdout. Ketika dataset/model diubah dan dilatih ulang, Excel menjadi sumber hasil terbaru; tabel statis README perlu diperbarui.

## 9. Cara menjalankan dan menggunakan model

Jalankan terminal dari folder proyek. Lingkungan hasil sekarang memakai Python 3.10. Versi pustaka dipatok di requirements.txt.

```powershell
python -m pip install -r requirements.txt
python -m playwright install chromium
```

Chromium hanya diperlukan untuk scraping. Perintah kerja:

```powershell
# Ambil data baru; tidak diperlukan jika sumber mentah sudah tersedia.
python scraper.py --target 4000

# Cleaning, seleksi fitur lewat CV training, dua CSV, dan sheet audit Data.
python pipeline.py preprocessing

# Baca dua CSV bersih, latih empat skenario, simpan evaluasi dan model.
python pipeline.py latih

# Preprocessing dan training berurutan, tanpa scraping otomatis.
python pipeline.py semua
```

Sumber khusus dapat dipilih melalui `python pipeline.py preprocessing --scrape data/nama_file.csv --broker data/nama_broker.xlsx`. Broker juga dapat berupa CSV, tetapi satu sumber broker saja yang dibaca.

Tutup CSV/Excel jika sedang dibuka di aplikasi yang mengunci file. Tahap preprocessing kini juga menjalankan CV seleksi atribut sehingga lebih lama daripada cleaning saja. Tahap ini tidak otomatis mengganti tabel Model atau melatih ulang `.joblib`. Jika baris, target, atau fitur berubah, lanjutkan dengan `latih` agar hasil konsisten. Penghapusan metadata alamat sebelumnya tidak mengubah prediksi; perubahan fitur seperti seleksi atribut dan penambahan rasio fisik memerlukan training ulang.

Contoh pemanggilan model:

```python
import joblib
import numpy as np
import pandas as pd

model = joblib.load("hasil/model_xgboost_fe.joblib")
contoh = pd.read_csv("data/dataset_dengan_fe.csv").head(1)
prediksi_rupiah = np.expm1(model.predict(contoh))
print(prediksi_rupiah)
```

Contoh memakai baris yang sudah tersedia hanya untuk menunjukkan pemanggilan, bukan evaluasi baru. Untuk rumah baru, sediakan fitur fisik dan furnished, serta kecamatan, jarak, dan dua rasio fisik dengan definisi FE yang sama. Alamat dan harga target tidak diperlukan sebagai input prediksi; kolom tambahan dataset yang tidak dipilih ColumnTransformer diabaikan.

## 10. Peta fungsi kode

| Fungsi | Lokasi | Tujuan |
|---|---|---|
| run_scraper, process_listing | scraper.py | Mengatur pencarian/kuota dan ekstraksi atribut |
| load_sources | pipeline.py | Membaca sumber dan memeriksa kolom wajib, termasuk alamat sumber |
| deduplicate, outside_surabaya, is_nonresidential | pipeline.py | Menyaring identitas dan cakupan objek penelitian |
| extract_property_condition | pipeline.py | Audit cakupan kondisi; tidak otomatis menjadi fitur aktif |
| select_reliable_features | pipeline.py | Seleksi carport/fasilitas/kondisi dan paket FE fisik hanya pada training |
| normalize_attributes | pipeline.py | Mengubah nilai invalid menjadi kosong tanpa imputasi global |
| remove_outliers_iqr | pipeline.py | Menyaring gabungan outlier harga/luas pada skala log |
| map_text_to_kecamatan, get_nearest_kecamatan | pipeline.py | Membaca label administratif atau memperkirakan kecamatan |
| haversine_distance, engineer_distance | pipeline.py | Membentuk fitur kecamatan dan jarak |
| save_datasets, run_preprocessing | pipeline.py | Memilih schema akhir tanpa alamat dan menjalankan cleaning |
| build_grid_around, run_experiments | pipeline.py | Menyusun grid lokal dan menjalankan empat eksperimen |
| evaluate_model | pipeline.py | Mengembalikan prediksi ke rupiah dan menghitung metrik |
| write_sheets, write_model_results | pipeline.py | Menyimpan audit, hasil, parameter, dan riwayat dalam satu Excel |

## 11. Panduan menyusun Bab III untuk penulis atau AI

Gunakan README sebagai deskripsi implementasi, bukan sebagai pengganti rujukan ilmiah. Teori XGBoost, Random Forest, Haversine, IQR, dan evaluasi memerlukan referensi akademik yang benar-benar diperiksa. Jangan membuat sitasi atau menyatakan metode yang belum diimplementasikan.

Usulan susunan berikut dapat disesuaikan dengan format kampus:

| Bagian Bab III | Isi yang perlu ditulis | Acuan README |
|---|---|---|
| 3.1 Rancangan penelitian | Eksperimen kuantitatif regresi harga; XGBoost utama dan RF pembanding; empat skenario | 2, 7 |
| 3.2 Objek dan sumber data | Listing harga penawaran Surabaya, Rumah123 dan broker, unit observasi dan kuota pencarian | 3 |
| 3.3 Variabel penelitian | Harga sebagai target, fitur fisik/lokasi terpilih, metadata, satuan, alasan atribut ditolak | 5, 6 |
| 3.4 Pengumpulan data | Playwright, ekstraksi payload, round-robin, kuota wilayah, deduplikasi URL | 3 |
| 3.5 Preprocessing | Urutan filter, normalisasi, batas minimum, IQR log, imputasi hanya saat training | 4 |
| 3.6 Feature engineering | Kecamatan, Haversine, fallback, dua rasio fisik, penghapusan GPS setelah FE | 6 |
| 3.7 Pemodelan dan tuning | Split bersama, seleksi atribut dan paket FE, pipeline fold, ruang Randomized dan GridSearchCV | 7 |
| 3.8 Evaluasi | MAE/RMSE/R2 pada harga rupiah, MAPE tambahan, perbandingan empat skenario | 8 |
| 3.9 Implementasi dan reproduksibilitas | Pustaka, perintah, keluaran Excel dan model tersimpan | 1, 9, 10 |

Angka hasil skor serta pembahasannya biasanya ditempatkan di Bab IV; Bab III menjelaskan rancangan dan cara menghitungnya. Jumlah data proses dapat disesuaikan dengan pedoman kampus. Gunakan bentuk waktu yang sesuai: implementasi dan eksperimen ini sudah dijalankan.

**Batas klaim yang harus dipertahankan dalam naskah:**

- Ini prediksi harga penawaran, bukan harga transaksi yang sudah tervalidasi.
- Kuota pencarian tidak menjamin sampel representatif seluruh pasar atau seimbang pada setiap kecamatan.
- IQR masih global sebelum split. Imputasi/encoding berada dalam training fold, tetapi jangan mengklaim semua kemungkinan data leakage sudah dihilangkan.
- Deduplikasi URL belum menjamin satu rumah fisik hanya muncul satu kali jika dipasarkan banyak agen. Tidak ada group split berdasarkan identitas rumah.
- Holdout pernah dilihat dalam pengembangan; hasil ini belum validasi eksternal independen. Seleksi fasilitas dan tuning bukan nested CV; tidak ada uji signifikansi statistik atau interval kepercayaan yang dihitung.
- Jarak memakai pin/centroid dan satu titik acuan; bukan jarak jalan, waktu tempuh, atau analisis akses semua fasilitas publik.
- FE yang dibandingkan adalah penggantian GPS dengan kecamatan dan jarak, ditambah dua fitur fisik terpilih. Kondisi hanya diaudit dan tidak aktif pada revisi ini. Jangan mengklaim ada harga per m2, riwayat renovasi terverifikasi, atau fitur lain yang tidak diterapkan.
- Tuning mengoptimalkan error target log. R2 testing dipakai untuk laporan, bukan untuk memilih parameter atau mengganti seed hingga skor tinggi.
- Aplikasi web dan rekomendasi rumah sebanding belum dibuat dalam proyek ini. Jika dibahas sebagai rencana lanjutan, pisahkan dengan jelas dari implementasi yang sudah selesai.
