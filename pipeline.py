"""Pipeline rumah Surabaya: preprocessing dan eksperimen dalam satu script.

Pemakaian:
    python pipeline.py preprocessing  # Membuat dua CSV bersih.
    python pipeline.py latih          # Membandingkan empat skenario model.
    python pipeline.py semua          # Preprocessing lalu pelatihan.

Scraping terpisah di scraper.py. Semua hasil tabel masuk satu buku Excel,
bukan beberapa CSV/JSON/log. Imputasi dan pemilihan fitur dipelajari hanya dari data training.
"""
import argparse
import json
import math
import os
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "hasil"


# =============================================================================
# 1. PENYIMPANAN HASIL: satu Excel berisi tabel yang berbeda pada tiap sheet.
# =============================================================================
def audit_table(report):
    """Ratakan ringkasan cleaning menjadi tabel agar mudah dibaca di Excel."""
    rows = []
    def collect(section, key, value):
        if isinstance(value, dict):
            for child, item in value.items():
                collect(section, f"{key}.{child}" if key else child, item)
        elif isinstance(value, list):
            for number, item in enumerate(value, 1):
                collect(section, f"{key}[{number}]", item)
        else:
            rows.append({"Bagian": section, "Keterangan": key, "Nilai": value})
    for section, value in report.items():
        collect(section, "", value)
    return pd.DataFrame(rows)


def write_sheets(updates):
    """Perbarui sheet yang diminta, pertahankan sheet hasil lainnya."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "hasil_penelitian.xlsx"
    sheets = pd.read_excel(path, sheet_name=None, keep_default_na=False) if path.exists() else {}
    sheets.update(updates)
    # Urutan tetap: pengguna langsung melihat hasil model saat membuka Excel.
    order = ["Model", "Parameter", "Data", "Importance", "Riwayat"]
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name in order:
            if name not in sheets:
                continue
            frame = sheets[name]
            frame.to_excel(writer, sheet_name=name, index=False)
            sheet = writer.sheets[name]
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
            for cells in sheet.columns:
                width = max(len(str(cell.value or "")) for cell in cells)
                sheet.column_dimensions[cells[0].column_letter].width = min(65, max(15, width + 2))
    print(f"Hasil tersimpan: {path}")


def write_model_results(results, parameters):
    """Simpan metrik, parameter terbaik, dan angka pembanding tanpa file tambahan."""
    path = OUTPUT_DIR / "hasil_penelitian.xlsx"
    history = pd.DataFrame(columns=["Versi"])
    if path.exists():
        with pd.ExcelFile(path) as workbook:
            if "Riwayat" in workbook.sheet_names:
                history = pd.read_excel(workbook, sheet_name="Riwayat")
    version = "seleksi_atribut_dan_fe_fisik"
    history = history.loc[history.Versi.ne(version)].copy()
    current = results.copy()
    current.insert(0, "Versi", version)
    data = pd.read_csv(DATA_DIR / "dataset_dengan_fe.csv")
    current["Jumlah Data"] = len(data)
    current["Harga Min"] = data.harga.min()
    current["Harga Max"] = data.harga.max()
    history = pd.concat([history, current], ignore_index=True)
    parameter_rows = [
        {"Skenario": scenario, "Parameter": parameter, "Nilai": json.dumps(value)}
        for scenario, values in parameters.items() for parameter, value in values.items()
    ]
    write_sheets({"Model": results, "Parameter": pd.DataFrame(parameter_rows), "Riwayat": history})


# =============================================================================
# 2. LOKASI: alamat asli untuk manusia; kecamatan dan jarak untuk fitur FE.
# =============================================================================
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
    """Baca label administratif eksplisit, bukan menebak dari nama kompleks.

    Bagian alamat yang persis nama kecamatan diprioritaskan. Nama jalan seperti
    Raya Kenjeran tidak otomatis berarti Kecamatan Kenjeran.
    """
    text = str(text).strip().lower()
    aliases = {k.lower(): k for k in KECAMATAN_CENTROIDS}
    aliases.update({"gununganyar": "Gunung Anyar", "pabean cantikan": "Pabean Cantian"})
    for part in re.split(r"[,;]", text):
        label = re.sub(r"^kec(?:amatan)?\.?\s+", "", part.strip())
        if label in aliases:
            return aliases[label]
    for alias, district in aliases.items():
        if re.search(r"\bkec(?:amatan)?\.?\s+" + re.escape(alias) + r"\b", text):
            return district
    return None


# =============================================================================
# 3. PREPROCESSING: sumber -> cleaning -> dua CSV dengan baris yang sama.
# =============================================================================
SIZE_PRICE_COLS = ("harga", "luas_tanah", "luas_bangunan")
COMMON_COLS = [
    "judul_listing", "harga", "luas_tanah", "luas_bangunan", "kamar_tidur",
    "kamar_mandi", "lantai", "carport", "furnished", "keamanan", "taman",
]
# Alamat hanya diperlukan pada sumber dan proses FE, tidak diekspor ke CSV bersih.
BASE_COLUMNS = COMMON_COLS + ["latitude", "longitude", "sumber_data"]
FE_PHYSICAL_COLUMNS = ["rasio_bangunan_tanah", "luas_bangunan_per_lantai"]
FE_COLUMNS = COMMON_COLS + ["kecamatan", "jarak_ke_pusat_kota"] + FE_PHYSICAL_COLUMNS + ["sumber_data"]
NEIGHBOR_PATTERN = r"\b(?:Mojokerto|Sidoarjo|Gresik|Lamongan|Bangkalan|Mojosari|Krian)\b"
AUCTION_PATTERN = r"\b(?:lelang|sita(?:an)?|disita|penyitaan|eksekusi)\b"


def load_sources(scrape_path=None, broker_path=None):
    """Baca satu CSV Rumah123 dan satu sumber broker, bukan Excel + CSV broker sekaligus."""
    if scrape_path is None:
        candidates = sorted(DATA_DIR.glob("rumah123_surabaya_raw_*.csv"))
        if not candidates:
            raise FileNotFoundError("CSV scraping tidak ditemukan. Jalankan scraper.py dahulu.")
        scrape_path = candidates[-1]
    scrape_path = Path(scrape_path)
    broker_path = Path(broker_path or DATA_DIR / "data_broker_surabaya.xlsx")
    scrape = pd.read_csv(scrape_path)
    broker = (pd.read_excel(broker_path) if broker_path.suffix.lower() == ".xlsx"
              else pd.read_csv(broker_path))
    if "wilayah_scraping" not in scrape:
        print("PERINGATAN: wilayah pencarian tidak tercatat pada data lama.")
        scrape["wilayah_scraping"] = "Tidak tercatat (data lama)"
    broker["wilayah_scraping"] = "Broker (bukan scraping)"
    frame = pd.concat([scrape, broker], ignore_index=True)
    required = set(BASE_COLUMNS + ["alamat_teks", "url_listing"]) - {"kondisi_properti"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Kolom sumber belum lengkap: {sorted(missing)}")
    print(f"Input scraping: {scrape_path}; broker: {broker_path} ({len(broker)} baris)")
    return frame, scrape_path, broker_path, len(broker)


def deduplicate(df):
    """URL kosong tidak dianggap satu identitas; bedakan unit broker lewat atributnya."""
    urls = df["url_listing"].fillna("").astype(str).str.strip()
    has_url = urls.ne("")
    keyed = df.loc[has_url].assign(url_listing=urls[has_url]).drop_duplicates("url_listing")
    keys = ["judul_listing", "alamat_teks", "harga", "luas_tanah",
            "luas_bangunan", "kamar_tidur", "kamar_mandi", "sumber_data"]
    unkeyed = df.loc[~has_url].drop_duplicates(keys)
    return pd.concat([keyed, unkeyed]).sort_index().copy()


def outside_surabaya(row):
    """Tolak lokasi luar; pengecualian harus didukung alamat administratif Surabaya."""
    title, address = str(row.get("judul_listing", "")), str(row.get("alamat_teks", ""))
    if not re.search(NEIGHBOR_PATTERN, title + " " + address, re.I):
        return False
    districts = "|".join(re.escape(k) for k in list(KECAMATAN_CENTROIDS) + ["Gununganyar", "Pabean Cantikan"])
    local_address = re.search(
        r"\b(?:" + districts + r")\s*,\s*(?:Kota\s+)?Surabaya\b", address, re.I
    )
    foreign_address = re.search(NEIGHBOR_PATTERN, address, re.I)
    return not (local_address and not foreign_address)


def is_nonresidential(title):
    """Saring jenis properti eksplisit; kata dekat/cocok/untuk bukan bukti jenis.

    Hanya klausa sebelum penawaran potensi/proksimitas diperiksa. Aturan ini
    konservatif: rumah dengan jumlah kamar besar saja tidak otomatis dibuang.
    """
    text = str(title).lower()
    text = re.split(r"\b(?:cocok|dekat|untuk|utk|unt|bisa|potensi|ada)\b", text)[0]
    patterns = [
        r"\brumah\s+(?:kos(?:t)?(?:-kosan)?|homestay|gudang)\b",
        r"\bkos(?:t)?(?:-kosan)?\s+(?:aktif|eksklusif|investment)\b",
        r"\beks\s+penginapan\b",
        r"^(?:(?:di\s*jual|jual|sewa)\s*(?:/\s*jual)?\s+)?(?:hotel|apartemen|ruko|gedung)\b",
    ]
    return any(re.search(pattern, text) for pattern in patterns)


def extract_property_condition(row):
    """Kondisi menurut klaim eksplisit sumber; bukan hasil inspeksi bangunan.

    Kolom kondisi sumber (jika tersedia) didahulukan. Data lama menggunakan
    judul saja, tanpa menebak kondisi dari harga, furnished, atau nama lokasi.
    Tidak diketahui dipertahankan sebagai kategori, bukan diisi modus Baik.
    """
    labels = {"baru": "Baru", "baik": "Terawat", "bagus": "Terawat",
              "bagus sekali": "Terawat", "terawat": "Terawat",
              "butuh renovasi": "Perlu renovasi", "perlu renovasi": "Perlu renovasi",
              "sedang renovasi": "Sedang renovasi", "baru renovasi": "Direnovasi",
              "baru direnovasi": "Direnovasi", "direnovasi": "Direnovasi"}
    raw = row.get("kondisi_properti", "")
    if pd.notna(raw) and str(raw).strip().lower() in labels:
        return labels[str(raw).strip().lower()], "kolom_sumber", str(raw)
    title = re.sub(r"\s+", " ", str(row.get("judul_listing", "")).lower()).strip()
    # Negasi membuat klaim tidak cukup pasti. 'Tidak terawat' bukan bukti
    # tingkat kerusakan tertentu, dan 'belum renovasi' bukan berarti rusak.
    if re.search(r"\b(?:tidak|tak|belum|bukan|tanpa)\s+(?:\w+\s+){0,2}(?:terawat|baru|renov\w*|bagus|baik)\b", title):
        return "Tidak diketahui", "judul_ambigu", ""
    patterns = [
        ("Perlu renovasi", r"\b(?:butuh|perlu|memerlukan)\s+(?:di\s*)?renov(?:asi)?\b"),
        ("Sedang renovasi", r"\b(?:sedang|proses|on progress)\s+(?:di\s*)?renov(?:asi)?\b"),
        ("Direnovasi", r"\b(?:baru|selesai|sudah)\s+(?:di\s*)?renov(?:asi)?\b"),
        ("Baru", r"\b(?:rumah\s+baru(?!\s+(?:di\s*)?renov)|baru\s+gress|brand\s+new|bangunan\s+baru)\b"),
    ]
    matches = [(label, re.search(pattern, title)) for label, pattern in patterns]
    matches = [(label, match.group(0)) for label, match in matches if match]
    if len(matches) == 1:
        return matches[0][0], "judul_listing", matches[0][1]
    if len(matches) > 1:
        return "Tidak diketahui", "judul_ambigu", ""
    good = re.search(r"\b(?:terawat|kondisi\s+(?:rumah\s+)?(?:bagus|baik|prima))\b", title)
    if good:
        return "Terawat", "judul_listing", good.group(0)
    return "Tidak diketahui", "tidak_tersedia", ""


def normalize_attributes(df):
    """Validasi tipe/kategori tanpa menghitung median/modus seluruh dataset.

    Nilai tidak masuk akal dijadikan kosong, bukan ditebak koreksinya.
    Batas berikut adalah aturan operasional untuk rumah tinggal penelitian ini.
    """
    df = df.copy()
    numeric = list(SIZE_PRICE_COLS) + ["kamar_tidur", "kamar_mandi", "lantai",
                                      "carport", "keamanan", "taman", "latitude", "longitude"]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    for col in ["kamar_tidur", "kamar_mandi", "lantai", "carport"]:
        invalid = (df[col] < 0) | (df[col] % 1 != 0)
        df.loc[invalid, col] = np.nan
    # Lebih dari 6 lantai diperlakukan sebagai entri yang perlu verifikasi.
    df.loc[(df.lantai < 1) | (df.lantai > 6), "lantai"] = np.nan
    # Parkir perlu sedikitnya 10 m2 per mobil; ini deteksi konservatif salah input.
    df.loc[df.carport * 10 > df.luas_tanah, "carport"] = np.nan
    for col, min_area in [("kamar_tidur", 6), ("kamar_mandi", 2)]:
        df.loc[df[col] * min_area > df.luas_bangunan, col] = np.nan
    for col in ["keamanan", "taman"]:
        df.loc[~df[col].isin([0, 1]), col] = np.nan
    valid_gps = df.latitude.between(-7.5, -7.1) & df.longitude.between(112.5, 112.9)
    df.loc[~valid_gps, ["latitude", "longitude"]] = np.nan
    categories = {"unfurnished": "Unfurnished", "semi furnished": "Semi Furnished",
                  "furnished": "Furnished", "fully furnished": "Furnished"}
    df["furnished"] = df.furnished.astype("string").str.strip().str.lower().map(categories)
    return df


def remove_outliers_iqr(df, columns=SIZE_PRICE_COLS):
    """Hitung batas di skala log; gabungkan mask dan simpan nilai asli, bukan log."""
    mask = pd.Series(False, index=df.index)
    for col in columns:
        values = np.log1p(df[col])
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = (values < lower) | (values > upper)
        print(f"IQR log {col}: {outliers.sum()} terdeteksi; "
              f"Q1={q1:.4f}, Q3={q3:.4f}, batas=[{lower:.4f}, {upper:.4f}]")
        mask |= outliers
    print(f"IQR log: {mask.sum()} baris unik dihapus.")
    return df.loc[~mask].copy()


def engineer_distance(df):
    """Ekstrak kecamatan dan jarak garis lurus; alamat asli tidak dimodifikasi.

    Kecamatan: label alamat > centroid GPS terdekat (perkiraan) > tidak diketahui.
    Jarak: GPS tersedia > centroid kecamatan > kosong. Tidak ada kecamatan default.
    """
    distances, districts, origins = [], [], []
    for _, row in df.iterrows():
        lat, lon = row["latitude"], row["longitude"]
        district = map_text_to_kecamatan(row["alamat_teks"])
        origin = "label_alamat"
        if district is None:
            district = get_nearest_kecamatan(lat, lon)
            origin = "perkiraan_centroid_GPS" if district else "tidak_diketahui"
        districts.append(district or "Tidak diketahui")
        origins.append(origin)
        if get_nearest_kecamatan(lat, lon) is None:
            lat, lon = KECAMATAN_CENTROIDS.get(district, (np.nan, np.nan))
        distances.append(haversine_distance(lat, lon, PUSAT_KOTA_LAT, PUSAT_KOTA_LON)
                         if pd.notna(lat) and pd.notna(lon) else np.nan)
    result = df.copy()
    result["kecamatan"] = districts
    result["jarak_ke_pusat_kota"] = distances
    # Rasio tidak memakai harga; luas per lantai adalah rata-rata, bukan tapak terukur.
    result["rasio_bangunan_tanah"] = result.luas_bangunan / result.luas_tanah.where(result.luas_tanah > 0)
    result["luas_bangunan_per_lantai"] = result.luas_bangunan / result.lantai.where(result.lantai > 0)
    result["asal_kecamatan"] = origins  # Audit saja; tidak masuk dataset/model.
    return result, result["kecamatan"]


def save_datasets(df, output_dir, selection):
    """Pilih schema eksplisit agar metadata scraping tidak bocor menjadi fitur."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / "dataset_tanpa_fe.csv", output_dir / "dataset_dengan_fe.csv"]
    # Periksa kedua file dahulu: Excel yang mengunci satu file tidak boleh
    # menyebabkan hanya satu dataset diperbarui dan pasangan baris tidak cocok.
    for path in paths:
        if path.exists():
            with path.open("r+b"):
                pass
    base_columns, fe_columns = selection["kolom_baseline"], selection["kolom_fe"]
    df[base_columns].to_csv(paths[0], index=False)
    df[fe_columns].to_csv(paths[1], index=False)
    print(f"Dataset tersimpan: {len(df)} baris, {len(base_columns)} kolom tanpa FE, "
          f"{len(fe_columns)} kolom dengan FE; folder {output_dir}")


def run_preprocessing(scrape_path=None, broker_path=None, output_dir=None):
    """Jalankan seluruh tahap berurutan dan tulis audit yang dapat dilaporkan."""
    output_dir = Path(output_dir or DATA_DIR)
    df, scrape_path, broker_path, broker_count = load_sources(scrape_path, broker_path)
    stages = {"setelah_gabung": len(df)}

    # 1-4. Sumber, deduplikasi, lokasi, lalu lelang (urutan memengaruhi hitungan audit).
    df = deduplicate(df)
    stages["setelah_dedup"] = len(df)
    outside = df.apply(outside_surabaya, axis=1)
    examples = df.loc[outside, "judul_listing"].head(10).tolist()
    print(f"Filter non-Surabaya: {outside.sum()} baris dihapus.")
    for title in examples:
        print(f"  - {title}")
    df = df.loc[~outside].copy()
    stages["setelah_filter_non_surabaya"] = len(df)
    auction = df["judul_listing"].fillna("").str.contains(AUCTION_PATTERN, case=False, regex=True)
    print(f"Filter lelang: {auction.sum()} baris dihapus.")
    df = df.loc[~auction].copy()
    stages["setelah_filter_lelang"] = len(df)
    if df.empty:
        raise ValueError("Tidak ada listing setelah filter lokasi/lelang.")

    commercial = df.judul_listing.map(is_nonresidential)
    commercial_examples = df.loc[commercial, "judul_listing"].tolist()
    df = df.loc[~commercial].copy()
    stages["setelah_filter_jenis_properti"] = len(df)
    before_normalize = df.copy()
    df = normalize_attributes(df)
    conditions = df.apply(extract_property_condition, axis=1)
    df["kondisi_properti"] = conditions.map(lambda item: item[0])
    df["asal_kondisi"] = conditions.map(lambda item: item[1])
    df["bukti_kondisi"] = conditions.map(lambda item: item[2])
    changed = {col: int((before_normalize[col].notna() & df[col].isna()).sum())
               for col in df.columns if col in before_normalize}
    stages["setelah_normalisasi"] = len(df)
    # Batas populasi dan IQR diterapkan sebelum split; imputasi dilakukan di CV.
    sane = (df.harga >= 100_000_000) & (df.luas_tanah >= 15) & (df.luas_bangunan >= 15)
    print(f"Hard sanity check: {(~sane).sum()} baris dihapus.")
    df = df.loc[sane].copy()
    stages["setelah_sanity_check"] = len(df)
    df = remove_outliers_iqr(df)
    stages["setelah_outlier_removal"] = len(df)
    if df.empty:
        raise ValueError("Tidak ada listing setelah penghapusan outlier.")

    # 8-9. Kecamatan dan jarak menjadi FE; identitas dan urutan baris kedua CSV tetap sama.
    df, district_audit = engineer_distance(df)
    selection = select_reliable_features(df)
    save_datasets(df, output_dir, selection)
    stages["final"] = len(df)
    report = {
        "input_scraping": str(scrape_path.resolve()), "input_broker": str(broker_path.resolve()),
        "output_dataset": str(output_dir.resolve()), "tahapan": stages,
        "contoh_non_surabaya_dihapus": examples,
        "jenis_properti_dihapus": commercial_examples,
        "nilai_invalid_dikosongkan": changed,
        "missing_final_sebelum_imputasi_training": df[BASE_COLUMNS].isna().sum().to_dict(),
        "asal_kecamatan": df.asal_kecamatan.value_counts().to_dict(),
        "seleksi_atribut_training": selection,
        "kondisi_properti": df.kondisi_properti.value_counts().to_dict(),
        "asal_kondisi": df.asal_kondisi.value_counts().to_dict(),
        "contoh_bukti_kondisi": df.loc[df.kondisi_properti.ne("Tidak diketahui")]
            .groupby("kondisi_properti").head(5)[["judul_listing", "kondisi_properti", "asal_kondisi", "bukti_kondisi"]]
            .to_dict(orient="records"),
        "dihapus_sanity": stages["setelah_normalisasi"] - stages["setelah_sanity_check"],
        "dihapus_iqr_log": stages["setelah_sanity_check"] - len(df),
        "wilayah_scraping": df.wilayah_scraping.fillna("Tidak tercatat").value_counts().to_dict(),
        "kecamatan_audit": district_audit.value_counts().reindex(list(KECAMATAN_CENTROIDS) + ["Tidak diketahui"], fill_value=0).sort_values(ascending=False).to_dict(),
        "sumber_data": df.sumber_data.value_counts().to_dict(),
        "rentang": {c: {"min": float(df[c].min()), "max": float(df[c].max())} for c in SIZE_PRICE_COLS},
        "broker_input": broker_count,
        "broker_final": int(df.sumber_data.str.contains("broker", case=False, na=False).sum()),
    }
    write_sheets({"Data": audit_table(report)})
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return df[selection["kolom_fe"]].copy()


# =============================================================================
# 4. EKSPERIMEN: baseline dan FE untuk Random Forest serta XGBoost.
# =============================================================================
PARAM_SPECS = {
    "n_estimators": {"step": 25, "min": 50, "max": None, "type": int},
    "max_depth": {"step": 1, "min": 1, "max": None, "type": int},
    "min_samples_split": {"step": 1, "min": 2, "max": None, "type": int},
    "min_samples_leaf": {"step": 1, "min": 1, "max": None, "type": int},
    "learning_rate": {"step": 0.01, "min": 0.01, "max": None, "type": float, "decimals": 3},
    "subsample": {"step": 0.05, "min": 0.5, "max": 1.0, "type": float, "decimals": 2},
    "colsample_bytree": {"step": 0.05, "min": 0.5, "max": 1.0, "type": float, "decimals": 2},
}


def build_grid_around(best_params, prefix, param_specs=PARAM_SPECS):
    """Membuat grid_params variasi di sekitar kandidat terbaik RandomizedSearchCV (coarse-to-fine search)."""
    grid = {}
    for key, val in best_params.items():
        clean_name = key.replace(f"{prefix}__", "")
        if val is None:
            grid[key] = [None]
        elif clean_name in param_specs:
            spec = param_specs[clean_name]
            step = spec["step"]
            min_val = spec.get("min")
            max_val = spec.get("max")

            if spec["type"] == int:
                candidates = [val - step, val, val + step]
                if min_val is not None:
                    candidates = [max(min_val, c) for c in candidates]
                if max_val is not None:
                    candidates = [min(max_val, c) for c in candidates]
                grid[key] = sorted(list(set(candidates)))
            else:
                decimals = spec.get("decimals", 2)
                candidates = [
                    round(val - step, decimals),
                    round(val, decimals),
                    round(val + step, decimals)
                ]
                if min_val is not None:
                    candidates = [max(min_val, c) for c in candidates]
                if max_val is not None:
                    candidates = [min(max_val, c) for c in candidates]
                grid[key] = sorted(list(set(candidates)))
        else:
            grid[key] = [val]
    return grid


def calculate_mape(y_true, y_pred):
    """Menghitung Mean Absolute Percentage Error (%)."""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# --- Evaluasi performa model pada skala Rupiah asli (R2, MAE, RMSE, MAPE) ---
def evaluate_model(model, X_test, y_test, is_log_target=True):
    """Menghitung metrik performa model pada data uji."""
    preds = model.predict(X_test)
    if is_log_target:
        preds = np.expm1(preds)
    return {
        "R2 Score": round(r2_score(y_test, preds), 4),
        "MAE (Juta Rp)": round(mean_absolute_error(y_test, preds) / 1_000_000, 2),
        "RMSE (Juta Rp)": round(np.sqrt(mean_squared_error(y_test, preds)) / 1_000_000, 2),
        "MAPE (%)": round(calculate_mape(y_test, preds), 2)
    }


def make_preprocessor(numeric, categories):
    """Imputasi dan encoding di-fit ulang hanya pada bagian training setiap fold."""
    return ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("encode", OneHotEncoder(handle_unknown="ignore"))]), categories)
    ])


def select_reliable_features(df):
    """Seleksi kandidat terbatas pada 80% training; holdout tidak ikut dinilai.

    Kondisi dengan cakupan <50% tidak layak menjadi atribut utama menurut
    kebijakan kelengkapan penelitian ini. Tetap diuji sebagai pembanding audit.
    Ambang ini adalah keputusan operasional, bukan kaidah statistik universal.
    """
    idx, _ = train_test_split(np.arange(len(df)), test_size=0.20, random_state=42)
    train = df.iloc[idx]
    target = np.log1p(train.harga)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    physical = ["luas_tanah", "luas_bangunan", "kamar_tidur", "kamar_mandi", "lantai"]
    condition_coverage = float(train.kondisi_properti.ne("Tidak diketahui").mean())
    rows = []

    def score(numeric, categories):
        model = Pipeline([("prep", make_preprocessor(numeric, categories)),
                          ("xgb", XGBRegressor(n_estimators=200, max_depth=6,
                           learning_rate=0.05, random_state=42, n_jobs=1))])
        folds = -cross_val_score(model, train, target, cv=cv,
                                 scoring="neg_mean_squared_error", n_jobs=-1)
        return float(folds.mean()), float(folds.std(ddof=1))

    for condition, facilities, parking in [(False, False, False), (False, False, True),
                                          (False, True, False), (False, True, True),
                                          (True, True, True)]:
        numeric = physical + (["carport"] if parking else [])
        numeric += (["keamanan", "taman"] if facilities else []) + ["latitude", "longitude"]
        categories = ["furnished"] + (["kondisi_properti"] if condition else [])
        mean, std = score(numeric, categories)
        rows.append(dict(kondisi=condition, fasilitas=facilities, carport=parking,
                         mse_log=mean, sd_fold=std,
                         layak=not condition or condition_coverage >= 0.5))
    best = min((r for r in rows if r["layak"]), key=lambda r: r["mse_log"])
    # Kondisi dapat dipakai kelak bila cakupan membaik dan kandidatnya menang.
    common = [c for c in COMMON_COLS if (best["fasilitas"] or c not in ("keamanan", "taman"))
              and (best["carport"] or c != "carport")]
    if best["kondisi"]:
        common.insert(common.index("furnished") + 1, "kondisi_properti")
    base_columns = common + ["latitude", "longitude", "sumber_data"]
    fe_numeric = [c for c in common if c not in ("judul_listing", "harga", "furnished", "kondisi_properti")]
    fe_numeric += ["jarak_ke_pusat_kota"]
    fe_categories = ["furnished", "kecamatan"] + (["kondisi_properti"] if best["kondisi"] else [])
    physical_rows = []
    for use_physical in [False, True]:
        mean, std = score(fe_numeric + (FE_PHYSICAL_COLUMNS if use_physical else []), fe_categories)
        physical_rows.append(dict(fe_fisik=use_physical, mse_log=mean, sd_fold=std))
    selected_physical = min(physical_rows, key=lambda r: r["mse_log"])["fe_fisik"]
    fe_columns = common + ["kecamatan", "jarak_ke_pusat_kota"]
    fe_columns += (FE_PHYSICAL_COLUMNS if selected_physical else []) + ["sumber_data"]
    report = dict(cakupan_kondisi_training=condition_coverage, ambang_kondisi=0.5,
                  kandidat_baseline=rows, kandidat_fe_fisik=physical_rows,
                  pilihan=best, fe_fisik_dipakai=selected_physical,
                  kolom_baseline=base_columns, kolom_fe=fe_columns)
    print("Seleksi atribut training:", json.dumps(report), flush=True)
    return report


def run_experiments(data_dir=DATA_DIR):
    """Menjalankan ablation study dan coarse-to-fine hyperparameter tuning untuk Random Forest dan XGBoost."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df_tanpa = pd.read_csv(os.path.join(data_dir, "dataset_tanpa_fe.csv"))
    df_dengan = pd.read_csv(os.path.join(data_dir, "dataset_dengan_fe.csv"))

    # Bandingkan semua atribut bersama agar kedua skenario memakai baris identik.
    identity = [c for c in df_tanpa.columns if c not in ('latitude', 'longitude')]
    if not df_tanpa[identity].equals(df_dengan[identity]):
        raise ValueError('Kedua dataset harus berisi baris yang sama dalam urutan yang sama.')
    y = df_dengan["harga"]
    # --- Log transform target (np.log1p) untuk menstabilkan variansi harga ---
    y_log = np.log1p(y)

    # Schema CSV dipilih lewat CV training pada tahap preprocessing.
    metadata = {"judul_listing", "harga", "sumber_data"}
    cat_cols_base = [c for c in ("furnished", "kondisi_properti") if c in df_tanpa]
    cat_cols_fe = cat_cols_base + ["kecamatan"]
    num_cols_base = [c for c in df_tanpa if c not in metadata | set(cat_cols_base)]
    num_cols_fe = [c for c in df_dengan if c not in metadata | set(cat_cols_fe)]
    prep_tanpa = make_preprocessor(num_cols_base, cat_cols_base)
    prep_dengan = make_preprocessor(num_cols_fe, cat_cols_fe)

    # --- Train-test split (80% train, 20% test) ---
    idx_train, idx_test = train_test_split(np.arange(len(df_dengan)), test_size=0.20, random_state=42)
    y_test = y.iloc[idx_test]
    y_log_train = y_log.iloc[idx_train]

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    best_params = {}

    rf_dist = {
        "rf__n_estimators": [100, 150, 200, 250],
        "rf__max_depth": [10, 15, 20, None],
        "rf__min_samples_split": [2, 5, 10],
        "rf__min_samples_leaf": [1, 2, 4]
    }
    xgb_dist = {
        "xgb__n_estimators": [150, 200, 250, 300],
        "xgb__max_depth": [4, 6, 8],
        "xgb__learning_rate": [0.02, 0.05, 0.1],
        "xgb__subsample": [0.8, 0.9, 1.0],
        "xgb__colsample_bytree": [0.8, 1.0]
    }

    # --- Definisi 4 skenario ablation study (Random Forest vs XGBoost) ---
    scenarios = [
        ("Random Forest", "Tanpa FE (Baseline)", prep_tanpa, df_tanpa, "rf",
         RandomForestRegressor(random_state=42, n_jobs=1), rf_dist),
        ("Random Forest", "DENGAN Feature Engineering", prep_dengan, df_dengan, "rf",
         RandomForestRegressor(random_state=42, n_jobs=1), rf_dist),
        ("XGBoost", "Tanpa FE (Baseline)", prep_tanpa, df_tanpa, "xgb",
         XGBRegressor(random_state=42, n_jobs=1), xgb_dist),
        ("XGBoost", "DENGAN Feature Engineering", prep_dengan, df_dengan, "xgb",
         XGBRegressor(random_state=42, n_jobs=1), xgb_dist),
    ]

    # Simpan pipeline XGBoost dengan FE dengan nama eksplisit, bukan klaim selalu terbaik.
    xgb_fe_model = None

    for model_name, scen_name, prep, df_src, prefix, estimator, dist in scenarios:
        print(f"Mulai {model_name}: {scen_name}", flush=True)
        pipe = Pipeline([("prep", prep), (prefix, estimator)])

        # --- Tahap 1: Pencarian kasar (coarse search) via RandomizedSearchCV ---
        rand = RandomizedSearchCV(
            pipe, dist, n_iter=16, cv=cv, scoring="neg_mean_squared_error",
            random_state=42, n_jobs=-1
        )
        print("RandomizedSearchCV: 16 kandidat x 5 fold", flush=True)
        rand.fit(df_src.iloc[idx_train], y_log_train)
        bp = rand.best_params_

        # --- Tahap 2: Pencarian presisi (fine tuning) di sekitar kandidat terbaik ---
        grid_params = build_grid_around(bp, prefix)
        grid = GridSearchCV(pipe, grid_params, cv=cv, scoring="neg_mean_squared_error", n_jobs=-1)
        print(f"GridSearchCV: {grid_params}", flush=True)
        grid.fit(df_src.iloc[idx_train], y_log_train)

        fitted_model = grid.best_estimator_
        key_name = f"{model_name}_{scen_name}".replace(" ", "_")
        best_params[key_name] = {k.replace(f"{prefix}__", ""): v for k, v in grid.best_params_.items()}

        best_params[key_name]["fitur_numerik"] = list(fitted_model.named_steps["prep"].transformers_[0][2])
        best_params[key_name]["fitur_kategori"] = list(fitted_model.named_steps["prep"].transformers_[1][2])
        best_params[key_name]["cv_neg_mse_log"] = float(grid.best_score_)
        metrics = evaluate_model(fitted_model, df_src.iloc[idx_test], y_test, is_log_target=True)
        results.append({"Model": model_name, "Skenario": scen_name, **metrics})
        print(metrics, flush=True)

        if model_name == "XGBoost" and scen_name == "DENGAN Feature Engineering":
            xgb_fe_model = fitted_model

    # --- Ekspor pipeline XGBoost FE (.joblib), feature importance, dan hasil eksperimen ---
    if xgb_fe_model:
        joblib.dump(xgb_fe_model, os.path.join(OUTPUT_DIR, "model_xgboost_fe.joblib"))

        # Ekstraksi feature importance
        prep_step = xgb_fe_model.named_steps["prep"]
        xgb_step = xgb_fe_model.named_steps["xgb"]
        all_features = prep_step.get_feature_names_out()

        fi_df = pd.DataFrame({
            "Fitur": all_features,
            "Importance": xgb_step.feature_importances_
        }).sort_values(by="Importance", ascending=False)
        write_sheets({"Importance": fi_df})

    df_results = pd.DataFrame(results)
    write_model_results(df_results, best_params)
    print(df_results.to_string(index=False))
    return df_results


def main():
    """Satu pintu untuk dua tahap utama, tanpa mengulang scraping secara otomatis."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tahap", choices=["preprocessing", "latih", "semua"])
    parser.add_argument("--scrape", help="CSV mentah pilihan; default file terbaru dalam data/.")
    parser.add_argument("--broker", help="Excel/CSV broker pilihan; default Excel broker dalam data/.")
    args = parser.parse_args()
    if args.tahap in ("preprocessing", "semua"):
        run_preprocessing(args.scrape, args.broker)
    if args.tahap in ("latih", "semua"):
        run_experiments()


if __name__ == "__main__":
    main()
