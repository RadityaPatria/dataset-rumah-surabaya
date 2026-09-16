import json
import math
import os
import pandas as pd

DATA_DIR = "data"
PUSAT_KOTA_LAT = -7.2575
PUSAT_KOTA_LON = 112.7521

KECAMATAN_CENTROIDS = {
    "Tegalsari": (-7.2721, 112.7386),
    "Simokerto": (-7.2396, 112.7533),
    "Genteng": (-7.2589, 112.7483),
    "Bubutan": (-7.2514, 112.7314),
    "Gubeng": (-7.2798, 112.7584),
    "Gunung Anyar": (-7.3361, 112.7915),
    "Sukolilo": (-7.2885, 112.7845),
    "Tambaksari": (-7.2523, 112.7661),
    "Mulyorejo": (-7.2655, 112.7925),
    "Rungkut": (-7.3195, 112.7758),
    "Tenggilis Mejoyo": (-7.3186, 112.7583),
    "Benowo": (-7.2372, 112.6328),
    "Pakal": (-7.2425, 112.6072),
    "Asemrowo": (-7.2486, 112.6989),
    "Sukomanunggal": (-7.2725, 112.7086),
    "Tandes": (-7.2619, 112.6783),
    "Sambikerep": (-7.2798, 112.6489),
    "Lakarsantri": (-7.3105, 112.6472),
    "Wonokromo": (-7.2985, 112.7381),
    "Wonocolo": (-7.3182, 112.7386),
    "Wiyung": (-7.3092, 112.6845),
    "Karangpilang": (-7.3325, 112.7028),
    "Jambangan": (-7.3189, 112.7167),
    "Gayungan": (-7.3328, 112.7297),
    "Dukuh Pakis": (-7.2925, 112.7025),
    "Sawahan": (-7.2789, 112.7228),
    "Bulak": (-7.2355, 112.7981),
    "Kenjeran": (-7.2289, 112.7786),
    "Semampir": (-7.2214, 112.7525),
    "Pabean Cantian": (-7.2285, 112.7386),
    "Krembangan": (-7.2361, 112.7225)
}


def haversine_distance(lat1, lon1, lat2, lon2):
    """Menghitung jarak lengkung bumi antara dua koordinat (km)."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    return round(R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)), 2)


def get_nearest_kecamatan(lat, lon):
    """Mencari kecamatan terdekat berdasarkan jarak koordinat minimum."""
    if pd.isna(lat) or pd.isna(lon) or not (-7.5 <= lat <= -7.1 and 112.5 <= lon <= 112.9):
        return None
    return min(KECAMATAN_CENTROIDS.keys(), key=lambda k: haversine_distance(lat, lon, *KECAMATAN_CENTROIDS[k]))


def map_text_to_kecamatan(text):
    """Memetakan kata kunci lokasi perumahan ke nama kecamatan administratif."""
    t = str(text).lower()
    rules = [
        ("Sambikerep", ["citraland", "northwest", "buona vista", "stamford", "woodland", "palma", "bukit palma"]),
        ("Lakarsantri", ["lakarsantri", "crystal golf", "jeruk", "bangkingan"]),
        ("Wiyung", ["royal residence", "wisata bukit mas", "wiyung", "babatan", "dian istana", "graha sampurna"]),
        ("Dukuh Pakis", ["graha famili", "graha family", "darmo permai", "dukuh pakis", "hr muhammad", "mayjend sungkono"]),
        ("Sukomanunggal", ["satelit", "darmo harapan", "tanjungsari", "sukomanunggal", "simomulyo"]),
        ("Tandes", ["tandes", "manukan", "balongsari", "banjar sugihan"]),
        ("Benowo", ["grand pakuwon", "benowo", "sememi", "kandangan"]),
        ("Pakal", ["pakal", "babat jerawat", "pondok benowo"]),
        ("Mulyorejo", ["pakuwon city", "san antonio", "laguna", "florence", "mulyosari", "sutorejo", "mulyorejo"]),
        ("Sukolilo", ["grand eastern", "sukolilo", "keputih", "klampis", "araya", "semolowaru", "medokan semampir"]),
        ("Gubeng", ["manyar", "kertajaya", "gubeng", "dharmahusada", "pucang", "ngagel"]),
        ("Rungkut", ["rungkut", "wonorejo", "grand alana", "park sunrise", "pandugo", "medokan ayu", "nirwana"]),
        ("Gunung Anyar", ["gunung anyar", "purimas", "wiguna", "amphibi"]),
        ("Tenggilis Mejoyo", ["tenggilis", "kutisari", "kendangsari", "prapen"]),
        ("Tambaksari", ["tambaksari", "ploso", "pacar kembang", "gresikan"]),
        ("Wonokromo", ["darmo", "raya darmo", "wonokromo", "diponegoro", "jagir"]),
        ("Wonocolo", ["wonocolo", "siwalankerto", "jemursari", "sidosermo", "margorejo"]),
        ("Gayungan", ["gayungan", "menanggal", "injoko", "ketintang baru"]),
        ("Jambangan", ["jambangan", "karah", "ketintang", "kebonsari"]),
        ("Karangpilang", ["karangpilang", "kebraon", "kedurus", "mastrip"]),
        ("Sawahan", ["sawahan", "kupang", "banyu urip", "petemon", "putat"]),
        ("Tegalsari", ["tegalsari", "imam bonjol", "dr soetomo", "kartini", "pandegiling", "basuki rahmat"]),
        ("Genteng", ["genteng", "tunjungan", "embong", "ketabang"]),
        ("Bubutan", ["bubutan", "kramat gantung", "raden saleh"]),
        ("Simokerto", ["simokerto", "kapasan", "sidotopo"]),
        ("Bulak", ["pantai mentari", "kenjeran", "bulak", "sukolilo baru"]),
        ("Semampir", ["semampir", "ampel", "pegirian", "wonokusumo"]),
        ("Pabean Cantian", ["pabean", "cantian", "kembang jepun", "perak"]),
        ("Krembangan", ["krembangan", "morokrembangan", "perak barat"]),
        ("Asemrowo", ["asemrowo", "asrowo", "tanjung sari"])
    ]
    for kec, keywords in rules:
        if any(kw in t for kw in keywords):
            return kec
    return None


def remove_outliers_iqr(df, columns=["harga", "luas_tanah", "luas_bangunan"]):
    """Menghapus baris outlier menggunakan metode Interquartile Range (IQR)."""
    n_before = len(df)
    print("\n--- Tahap 4: Outlier Removal Menggunakan Metode Interquartile Range (IQR) ---")
    print(f"Jumlah baris sebelum outlier removal: {n_before}")

    outlier_mask = pd.Series(False, index=df.index)

    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)

        is_col_outlier = (df[col] < lower_bound) | (df[col] > upper_bound)
        count_outlier = is_col_outlier.sum()
        print(f"Outlier removal pada kolom {col}: {count_outlier} baris dihapus dari total {n_before} baris "
              f"(Q1={q1:,.0f}, Q3={q3:,.0f}, IQR={iqr:,.0f}, Batas Bawah={lower_bound:,.0f}, Batas Atas={upper_bound:,.0f})")

        outlier_mask = outlier_mask | is_col_outlier

    df_cleaned = df[~outlier_mask].copy()
    n_after = len(df_cleaned)
    total_removed = n_before - n_after
    print(f"Jumlah baris sesudah outlier removal: {n_after}")
    print(f"Total baris outlier yang dihapus: {total_removed} baris\n")

    return df_cleaned


def run_preprocessing():
    """Menggabungkan data, melakukan pembersihan, dan menghitung fitur rekayasa."""
    # --- Tahap 1: Load dan gabungkan data mentah Rumah123 dan broker developer ---
    df_scrape = pd.read_csv(os.path.join(DATA_DIR, "rumah123_surabaya_raw_2026-09-14.csv"))
    df_broker = pd.read_csv(os.path.join(DATA_DIR, "data_broker_surabaya_raw.csv"))

    df = pd.concat([df_scrape, df_broker], ignore_index=True)
    print(f"Tahap 1: Load data mentah selesai. Total gabungan: {len(df)} baris.")

    # --- Tahap 2: Hapus duplikat berdasarkan url_listing ---
    n_before_dup = len(df)
    df = df.drop_duplicates(subset=["url_listing"]).copy()
    print(f"Tahap 2: Hapus duplikat selesai. {n_before_dup - len(df)} duplikat dihapus, tersisa {len(df)} baris.")

    # --- Tahap 3: Missing value imputation (median untuk numerik, modus untuk furnished) ---
    print("\n--- Tahap 3: Missing Value Imputation ---")
    numeric_cols = ["kamar_tidur", "kamar_mandi", "lantai", "carport"]
    for col in numeric_cols:
        med_val = int(round(df[col].median()))
        null_count = df[col].isna().sum()
        df[col] = df[col].fillna(med_val).astype(int)
        print(f"Median {col} yang dipakai: {med_val} (mengisi {null_count} nilai kosong)")

    mode_furnished = df["furnished"].mode()[0]
    null_fur_count = df["furnished"].isna().sum()
    df["furnished"] = df["furnished"].fillna(mode_furnished)
    print(f"Modus furnished yang dipakai: {mode_furnished} (mengisi {null_fur_count} nilai kosong)")

    # --- Tahap 4: Outlier removal dengan IQR pada harga, luas_tanah, luas_bangunan ---
    df = remove_outliers_iqr(df, columns=["harga", "luas_tanah", "luas_bangunan"])

    # --- Tahap 5: Feature engineering ---
    # FE 1: Pemetaan dan standardisasi ke 31 kecamatan resmi Surabaya
    kecamatan_list = []
    for _, row in df.iterrows():
        kec = map_text_to_kecamatan(f"{row['alamat_teks']} {row['judul_listing']}")
        if not kec:
            kec = get_nearest_kecamatan(row["latitude"], row["longitude"])
        kecamatan_list.append(kec or "Sukolilo")
    df["kecamatan"] = kecamatan_list

    # FE 2: Rasio luas bangunan terhadap luas tanah
    df["rasio_bangunan_tanah"] = (df["luas_bangunan"] / df["luas_tanah"]).round(2)

    # FE 3: Total ruangan inti (kamar tidur + kamar mandi)
    df["total_ruangan"] = df["kamar_tidur"] + df["kamar_mandi"]

    # FE 4: Jarak ke pusat kota Surabaya (rumus Haversine)
    jarak_list = []
    for _, row in df.iterrows():
        lat, lon, kec = row["latitude"], row["longitude"], row["kecamatan"]
        if pd.notna(lat) and pd.notna(lon) and (-7.5 <= lat <= -7.1 and 112.5 <= lon <= 112.9):
            d = haversine_distance(lat, lon, PUSAT_KOTA_LAT, PUSAT_KOTA_LON)
        else:
            c_lat, c_lon = KECAMATAN_CENTROIDS.get(kec, (PUSAT_KOTA_LAT, PUSAT_KOTA_LON))
            d = haversine_distance(c_lat, c_lon, PUSAT_KOTA_LAT, PUSAT_KOTA_LON)
        jarak_list.append(d)
    df["jarak_ke_pusat_kota"] = jarak_list

    # --- Tahap 6: Simpan dataset ---
    cols_tanpa = [
        "judul_listing", "harga", "luas_tanah", "luas_bangunan", "kamar_tidur",
        "kamar_mandi", "lantai", "carport", "furnished", "keamanan", "taman",
        "latitude", "longitude", "alamat_teks", "sumber_data"
    ]
    df_tanpa = df[cols_tanpa].copy()
    df_tanpa.to_csv(os.path.join(DATA_DIR, "dataset_tanpa_fe.csv"), index=False)

    cols_dengan = [
        "judul_listing", "harga", "luas_tanah", "luas_bangunan", "kamar_tidur",
        "kamar_mandi", "lantai", "carport", "furnished", "keamanan", "taman",
        "latitude", "longitude", "kecamatan", "jarak_ke_pusat_kota",
        "rasio_bangunan_tanah", "total_ruangan", "alamat_teks", "sumber_data"
    ]
    df_dengan = df[cols_dengan].copy()
    df_dengan.to_csv(os.path.join(DATA_DIR, "dataset_dengan_fe.csv"), index=False)

    # Metadata untuk Web App
    metadata = {}
    for kec, (lat, lon) in KECAMATAN_CENTROIDS.items():
        sub = df[df["kecamatan"] == kec]
        metadata[kec] = {
            "latitude": lat,
            "longitude": lon,
            "total_listings": len(sub),
            "median_harga": int(sub["harga"].median()) if len(sub) > 0 else 2_000_000_000,
            "jarak_ke_pusat_km": round(haversine_distance(lat, lon, PUSAT_KOTA_LAT, PUSAT_KOTA_LON), 2)
        }
    with open(os.path.join(DATA_DIR, "metadata_kecamatan.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # --- Ringkasan Akhir Preprocessing ---
    print("=" * 60)
    print("RINGKASAN AKHIR PREPROCESSING")
    print("=" * 60)
    print(f"Jumlah baris dataset final: {len(df)}")
    print(f"Jumlah kolom dataset_tanpa_fe.csv: {df_tanpa.shape[1]} kolom")
    print(f"Jumlah kolom dataset_dengan_fe.csv: {df_dengan.shape[1]} kolom")
    print(f"Konsistensi jumlah baris: {len(df_tanpa) == len(df_dengan)} ({len(df_tanpa)} baris)")
    print("\nRentang Nilai (Min - Max) setelah Outlier Removal:")
    for col in ["harga", "luas_tanah", "luas_bangunan"]:
        print(f"- {col}: Min = {df[col].min():,.0f} | Max = {df[col].max():,.0f}")
    print("=" * 60)
    print(f"Preprocessing selesai. {len(df)} data bersih berhasil disimpan ke folder '{DATA_DIR}/'.")
    return df_dengan


if __name__ == "__main__":
    run_preprocessing()
