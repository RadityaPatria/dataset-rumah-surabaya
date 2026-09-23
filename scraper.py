"""Ambil listing unik dari lima wilayah dengan kuota yang seimbang.

CSV mentah mempertahankan atribut sumber; wilayah_scraping hanya label audit.
Pembersihan lelang, lokasi luar kota, dan outlier dilakukan di preprocessing.
"""
import argparse
from datetime import datetime
import json
import logging
import os
import random
import re
import sys
import time

import pandas as pd
from playwright.sync_api import sync_playwright

if sys.platform == "win32":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("ScraperRumah123")

SOURCE_TAG = "rumah123"

# --- Setup konfigurasi dan schema kolom data ---
CSV_COLUMNS = [
    "judul_listing",
    "harga",
    "luas_tanah",
    "luas_bangunan",
    "kamar_tidur",
    "kamar_mandi",
    "lantai",
    "carport",
    "furnished",
    "keamanan",
    "taman",
    "alamat_teks",
    "latitude",
    "longitude",
    "url_listing",
    "sumber_data",
    "wilayah_scraping"
]


# --- Helper fungsi pembersihan dan ekstraksi atribut listing ---
def parse_price(price_obj):
    """Mengekstrak harga dalam satuan Rupiah integer dari objek harga atau teks display."""
    if not price_obj:
        return None

    offer = price_obj.get("offer") if isinstance(price_obj, dict) else None
    if offer is not None:
        try:
            val = int(offer)
            if val > 0:
                return val
        except (ValueError, TypeError):
            pass

    display = price_obj.get("display") if isinstance(price_obj, dict) else str(price_obj or "")
    if not display:
        return None

    match = re.search(r'([\d.,]+)\s*(miliar|milyar|m|juta|jt)?', display, re.I)
    if match:
        number = match.group(1)
        unit = (match.group(2) or '').lower()
        if unit:
            if ',' in number:
                number = number.replace('.', '').replace(',', '.')
            elif re.fullmatch(r'\d{1,3}(?:\.\d{3})+', number):
                number = number.replace('.', '')
        else:
            number = number.replace('.', '').replace(',', '')
        try:
            value = float(number)
        except ValueError:
            return None
        multiplier = 1_000_000_000 if unit in ('miliar', 'milyar', 'm') else 1_000_000 if unit else 1
        return int(value * multiplier)
    return None


def parse_float_safe(val):
    """Konversi nilai numerik desimal secara aman."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, dict):
        val = val.get("value") or val.get("formattedValue")
    if not val:
        return None
    val_str = str(val).replace(",", ".").strip()
    match = re.search(r"[-+]?\d*\.?\d+", val_str)
    return float(match.group(0)) if match else None


def parse_int_safe(val):
    """Konversi nilai integer secara aman tanpa nilai default."""
    if val is None:
        return None
    if isinstance(val, int):
        return val
    if isinstance(val, dict):
        val = val.get("value") or val.get("formattedValue")
    if not val:
        return None
    val_str = str(val).strip()
    match = re.search(r"\d+", val_str)
    return int(match.group(0)) if match else None


def check_boolean_facility(listing, keywords):
    """Mengecek keberadaan kata kunci fasilitas pada atribut dan teks listing."""
    texts = [
        listing.get("title") or "",
        listing.get("shortDescription") or ""
    ]

    overview = listing.get("overview") or []
    if isinstance(overview, list):
        texts.extend([str(item) for item in overview])
    elif isinstance(overview, str):
        texts.append(overview)

    attrs = listing.get("attributes") or {}
    for key in ("residentialFacilities", "roomFacilities"):
        fac = attrs.get(key)
        if isinstance(fac, dict):
            texts.extend([fac.get("value") or "", fac.get("formattedValue") or ""])
        elif isinstance(fac, str):
            texts.append(fac)

    combined = " ".join(texts).lower()
    return int(any(kw.lower() in combined for kw in keywords))


def extract_listings_from_rsc(html):
    """Mengekstrak data JSON listing dari payload Next.js React Server Component."""
    matches = re.findall(r'self\.__next_f\.push\(\[1,\s*"(.*?)"\]\)', html)
    if not matches:
        return []

    full_stream = ""
    for m in matches:
        try:
            full_stream += json.loads('"' + m + '"')
        except Exception:
            full_stream += m

    listing_starts = list(re.finditer(r'\{"slug":"/properti/[^"]*hos\d+[^"]*"', full_stream))
    listings = []

    for m in listing_starts:
        pos = m.start()
        depth = 0
        in_string = False
        escape = False
        end = -1

        for i in range(pos, min(len(full_stream), pos + 35000)):
            ch = full_stream[i]
            if escape:
                escape = False
                continue
            if ch == '\\':
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if not in_string:
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break

        if end != -1:
            try:
                listings.append(json.loads(full_stream[pos:end]))
            except Exception:
                pass

    return listings


def process_listing(item):
    """Memvalidasi dan memetakan objek listing mentah ke schema 16 kolom standar."""
    slug = item.get("slug") or item.get("url") or ""
    if not slug:
        return None, "URL kosong"
    url_listing = slug if slug.startswith("http") else f"https://www.rumah123.com{slug}"

    harga = parse_price(item.get("price"))
    if not harga or harga <= 0:
        return None, "Harga tidak valid"

    attrs = item.get("attributes") or {}
    luas_tanah = parse_float_safe(attrs.get("landSize"))
    luas_bangunan = parse_float_safe(attrs.get("buildingSize"))
    if not luas_tanah or not luas_bangunan:
        return None, "Luas tanah atau bangunan kosong"

    furnished_val = attrs.get("furnishing") or attrs.get("furnished")
    if isinstance(furnished_val, dict):
        furnished_val = furnished_val.get("formattedValue") or furnished_val.get("value")

    locations = item.get("locations") or []
    address_parts = [loc.get("name") for loc in locations if isinstance(loc, dict) and loc.get("name")]
    alamat_teks = (item.get("location") or {}).get("text") or (", ".join(dict.fromkeys(address_parts)) if address_parts else (item.get("address") or "Surabaya"))

    coord = item.get("locationPin") or item.get("coordinate") or {}
    row = {
        "judul_listing": (item.get("title") or "").strip(),
        "harga": harga,
        "luas_tanah": luas_tanah,
        "luas_bangunan": luas_bangunan,
        "kamar_tidur": parse_int_safe(attrs.get("bedrooms")),
        "kamar_mandi": parse_int_safe(attrs.get("bathrooms")),
        "lantai": parse_int_safe(attrs.get("floors") or attrs.get("tier")),
        "carport": parse_int_safe(attrs.get("carports") or attrs.get("garages")),
        "furnished": str(furnished_val).strip() if furnished_val else None,
        "keamanan": check_boolean_facility(item, ["keamanan", "security", "satpam", "one gate", "cctv", "24 jam"]),
        "taman": check_boolean_facility(item, ["taman", "garden", "halaman", "backyard"]),
        "alamat_teks": alamat_teks,
        "latitude": parse_float_safe(coord.get("latitude") or coord.get("lat")),
        "longitude": parse_float_safe(coord.get("longitude") or coord.get("lon") or coord.get("lng")),
        "url_listing": url_listing,
        "sumber_data": SOURCE_TAG
    }
    return row, None


# Pembagian resmi: https://jdih.surabaya.go.id/peraturan/download/4588
REGIONS = {
    'Surabaya Timur': ['Gubeng', 'Gunung Anyar', 'Sukolilo', 'Tambaksari', 'Mulyorejo', 'Rungkut', 'Tenggilis Mejoyo'],
    'Surabaya Barat': ['Benowo', 'Pakal', 'Asemrowo', 'Sukomanunggal', 'Tandes', 'Sambikerep', 'Lakarsantri'],
    'Surabaya Utara': ['Bulak', 'Kenjeran', 'Semampir', 'Pabean Cantian', 'Krembangan'],
    'Surabaya Selatan': ['Wonokromo', 'Wonocolo', 'Wiyung', 'Karangpilang', 'Jambangan', 'Gayungan', 'Dukuh Pakis', 'Sawahan'],
    'Surabaya Pusat': ['Tegalsari', 'Simokerto', 'Genteng', 'Bubutan'],
}


def search_url(district, page_number):
    slug = district.lower().replace(' ', '-')
    if district == 'Gunung Anyar':
        slug = 'gununganyar'
    if district == 'Pabean Cantian':
        slug = 'pabean-cantikan'
    return f'https://www.rumah123.com/jual/surabaya/{slug}/rumah/?page={page_number}'


def run_scraper(target_count=4000, delay_min=3.5, delay_max=6.5, headless=True,
                max_pages=250):
    if target_count < 5 or not 0 <= delay_min <= delay_max or max_pages < 1:
        raise ValueError('Target minimal 5, delay valid, max_pages minimal 1.')
    os.makedirs('data', exist_ok=True)
    output_csv = os.path.join('data', f'rumah123_surabaya_raw_{datetime.now():%Y-%m-%d}_balanced.csv')
    quotas = {r: target_count // 5 + (i < target_count % 5) for i, r in enumerate(REGIONS)}
    counts = dict.fromkeys(REGIONS, 0)
    seen_urls = set()
    if os.path.exists(output_csv):
        existing = pd.read_csv(output_csv)
        if list(existing.columns) != CSV_COLUMNS or not existing['wilayah_scraping'].isin(REGIONS).all():
            raise ValueError('Schema/label wilayah CSV resume tidak valid.')
        seen_urls.update(existing['url_listing'].dropna())
        counts.update(existing['wilayah_scraping'].value_counts().to_dict())
        if any(counts[r] > quotas[r] for r in REGIONS):
            raise ValueError('Target lebih kecil dari data resume; gunakan target sebelumnya atau lebih besar.')
    else:
        pd.DataFrame(columns=CSV_COLUMNS).to_csv(output_csv, index=False)
    # Round-robin halaman antar wilayah DAN kecamatan agar satu kompleks tidak menghabiskan kuota.
    cursors = {(r, d): 1 for r, ds in REGIONS.items() for d in ds}
    empty = dict.fromkeys(cursors, 0)
    stopped = False
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=headless)
            context = browser.new_context(locale='id-ID', viewport={'width': 1366, 'height': 768})
            page = context.new_page()
            try:
                while any(counts[r] < quotas[r] for r in REGIONS) and not stopped:
                    attempted = False
                    for slot in range(max(map(len, REGIONS.values()))):
                        for region, districts in REGIONS.items():
                            if stopped or counts[region] >= quotas[region] or slot >= len(districts):
                                continue
                            district = districts[slot]
                            key = (region, district)
                            if cursors[key] > max_pages or empty[key] >= 3:
                                continue
                            attempted = True
                            url = search_url(district, cursors[key])
                            logger.info('%s | %s | %s', region, district, url)
                            try:
                                response = page.goto(url, timeout=60000, wait_until='domcontentloaded')
                                title = page.title().lower()
                                status = response.status if response else 0
                                if status in (403, 429) or any(w in title for w in ['just a moment', 'captcha', 'challenge']):
                                    logger.error('Akses dibatasi situs; scraping dihentikan. Kuota belum tentu terpenuhi.')
                                    stopped = True
                                    break
                                if status != 200 or '/jual/surabaya/' not in page.url or district.lower().split()[0] not in title:
                                    logger.warning('Halaman tidak cocok/HTTP %s: %s', status, page.url)
                                    empty[key] += 1
                                    cursors[key] += 1
                                    continue
                                raw = extract_listings_from_rsc(page.content())
                                rows = []
                                for item in raw:
                                    row, _ = process_listing(item)
                                    if row is None or row['url_listing'] in seen_urls:
                                        continue
                                    row['wilayah_scraping'] = region
                                    rows.append(row)
                                    seen_urls.add(row['url_listing'])
                                    counts[region] += 1
                                    if counts[region] >= quotas[region]:
                                        break
                                if rows:
                                    pd.DataFrame(rows)[CSV_COLUMNS].to_csv(output_csv, mode='a', header=False, index=False)
                                empty[key] = 0 if raw else empty[key] + 1
                                logger.info('%s: %d/%d (+%d)', region, counts[region], quotas[region], len(rows))
                            except Exception:
                                logger.exception('Gagal mengambil %s', url)
                                empty[key] += 1
                            cursors[key] += 1
                            time.sleep(random.uniform(delay_min, delay_max))
                    if not attempted:
                        break
            finally:
                browser.close()
    finally:
        for region in REGIONS:
            logger.info('HASIL %s: %d/%d (kekurangan %d)', region, counts[region], quotas[region], max(0, quotas[region]-counts[region]))
        logger.info('CSV gabungan: %s', output_csv)
    return output_csv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scraper Listing Rumah123 Surabaya")
    parser.add_argument("--target", type=int, default=4000, help="Jumlah target data")
    parser.add_argument("--delay-min", type=float, default=3.5, help="Delay minimal (detik)")
    parser.add_argument("--delay-max", type=float, default=6.5, help="Delay maksimal (detik)")
    parser.add_argument("--no-headless", action="store_true", help="Jalankan browser dengan UI")
    parser.add_argument("--max-pages", type=int, default=250, help="Batas halaman per kecamatan")
    args = parser.parse_args()

    run_scraper(
        target_count=args.target,
        delay_min=args.delay_min,
        delay_max=args.delay_max,
        headless=not args.no_headless,
        max_pages=args.max_pages
    )
