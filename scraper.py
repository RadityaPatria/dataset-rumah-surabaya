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
from playwright_stealth import Stealth

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("ScraperRumah123")

BASE_URL = "https://www.rumah123.com/jual/surabaya/rumah/"
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
    "sumber_data"
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

    cleaned = display.replace("Rp", "").replace(".", "").replace(",", ".").strip()
    match = re.search(r"([\d\.]+)\s*(miliar|m|juta|jt)?", cleaned, re.IGNORECASE)
    if match:
        num = float(match.group(1))
        unit = (match.group(2) or "").lower()
        if "miliar" in unit or unit == "m":
            return int(num * 1_000_000_000)
        elif "juta" in unit or unit == "jt":
            return int(num * 1_000_000)
        return int(num)
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
            full_stream += m.encode("utf-8").decode("unicode_escape")
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
    alamat_teks = ", ".join(dict.fromkeys(address_parts)) if address_parts else (item.get("address") or "Surabaya")

    coord = item.get("coordinate") or {}
    row = {
        "judul_listing": (item.get("title") or "").strip(),
        "harga": harga,
        "luas_tanah": luas_tanah,
        "luas_bangunan": luas_bangunan,
        "kamar_tidur": parse_int_safe(attrs.get("bedrooms")),
        "kamar_mandi": parse_int_safe(attrs.get("bathrooms")),
        "lantai": parse_int_safe(attrs.get("floors") or attrs.get("tier")),
        "carport": parse_int_safe(attrs.get("carports") or attrs.get("garages")),
        "furnished": str(furnished_val).strip() if furnished_val else "Unfurnished",
        "keamanan": check_boolean_facility(item, ["keamanan", "security", "satpam", "one gate", "cctv", "24 jam"]),
        "taman": check_boolean_facility(item, ["taman", "garden", "halaman", "backyard"]),
        "alamat_teks": alamat_teks,
        "latitude": parse_float_safe(coord.get("latitude") or coord.get("lat")),
        "longitude": parse_float_safe(coord.get("longitude") or coord.get("lon") or coord.get("lng")),
        "url_listing": url_listing,
        "sumber_data": SOURCE_TAG
    }
    return row, None


def run_scraper(target_count=3500, delay_min=3.5, delay_max=6.5, headless=True):
    """Menjalankan bot browser dengan modul stealth untuk scraping listing Rumah123."""
    os.makedirs("data", exist_ok=True)
    today_str = datetime.now().strftime("%Y-%m-%d")
    output_csv = os.path.join("data", f"rumah123_surabaya_raw_{today_str}.csv")

    seen_urls = set()
    total_valid = 0

    if os.path.exists(output_csv):
        try:
            existing_df = pd.read_csv(output_csv)
            if "url_listing" in existing_df.columns:
                seen_urls = set(existing_df["url_listing"].dropna().tolist())
                total_valid = len(existing_df)
                logger.info(f"Melanjutkan dari {total_valid} data yang sudah tersimpan di {output_csv}")
        except Exception as e:
            logger.warning(f"Gagal membaca file eksisting: {e}")

    if total_valid >= target_count:
        logger.info(f"Target {target_count} data sudah terpenuhi.")
        return output_csv

    current_page = (total_valid // 20) + 1
    max_pages = 250
    consecutive_empty = 0

    stealth = Stealth()

    # --- Setup browser Playwright dengan modul stealth ---
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=headless,
            args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-blink-features=AutomationControlled"]
        )
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
            viewport={"width": 1366, "height": 768},
            locale="id-ID"
        )
        page = context.new_page()
        stealth.apply_stealth_sync(page)

        try:
            # --- Loop scraping per halaman dan ekstraksi listing ---
            while total_valid < target_count and current_page <= max_pages:
                page_url = f"{BASE_URL}?page={current_page}"
                logger.info(f"Memproses halaman {current_page}: {page_url}")

                try:
                    response = page.goto(page_url, timeout=45000, wait_until="domcontentloaded")
                except Exception as e:
                    logger.warning(f"Percobaan 1 gagal ({e}), mencoba kembali...")
                    time.sleep(3)
                    try:
                        response = page.goto(page_url, timeout=60000, wait_until="domcontentloaded")
                    except Exception as e2:
                        logger.error(f"Halaman {current_page} dilewati karena timeout: {e2}")
                        current_page += 1
                        continue

                status = response.status if response else 0
                title = (page.title() or "").lower()

                if status in (403, 429) or any(w in title for w in ["just a moment", "captcha", "challenge"]):
                    logger.critical("Proteksi bot terdeteksi. Proses dihentikan demi keamanan.")
                    break

                raw_listings = extract_listings_from_rsc(page.content())
                if not raw_listings:
                    consecutive_empty += 1
                    if consecutive_empty >= 5:
                        logger.info("Listing tidak lagi ditemukan pada 5 halaman beruntun. Selesai.")
                        break
                    current_page += 1
                    time.sleep(random.uniform(delay_min, delay_max))
                    continue

                consecutive_empty = 0
                page_rows = []

                for raw_item in raw_listings:
                    row, _ = process_listing(raw_item)
                    if not row or row["url_listing"] in seen_urls:
                        continue

                    seen_urls.add(row["url_listing"])
                    page_rows.append(row)
                    total_valid += 1
                    if total_valid >= target_count:
                        break

                # --- Penyimpanan hasil scraping secara bertahap ke CSV ---
                if page_rows:
                    batch_df = pd.DataFrame(page_rows)[CSV_COLUMNS]
                    file_exists = os.path.exists(output_csv)
                    batch_df.to_csv(
                        output_csv,
                        mode="a" if file_exists else "w",
                        header=not file_exists,
                        index=False,
                        encoding="utf-8"
                    )

                logger.info(f"Halaman {current_page} selesai (+{len(page_rows)} data). Total: {total_valid}/{target_count}")
                if total_valid >= target_count:
                    break

                current_page += 1
                time.sleep(random.uniform(delay_min, delay_max))

        finally:
            browser.close()

    logger.info(f"Selesai. Total tersimpan: {total_valid} data di {output_csv}")
    return output_csv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scraper Listing Rumah123 Surabaya")
    parser.add_argument("--target", type=int, default=3500, help="Jumlah target data")
    parser.add_argument("--delay-min", type=float, default=3.5, help="Delay minimal (detik)")
    parser.add_argument("--delay-max", type=float, default=6.5, help="Delay maksimal (detik)")
    parser.add_argument("--no-headless", action="store_true", help="Jalankan browser dengan UI")
    args = parser.parse_args()

    run_scraper(
        target_count=args.target,
        delay_min=args.delay_min,
        delay_max=args.delay_max,
        headless=not args.no_headless
    )
