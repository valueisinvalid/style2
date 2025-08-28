import json
import re
import time
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import argparse


# ---------- helpers for link parsing ----------
def _normalize_url(line: str) -> str:
    s = line.strip()
    if s.startswith("//"):
        return "https:" + s
    if s.startswith("http://"):
        return "https://" + s[len("http://"):]
    return s


def parse_links_with_categories(path: Path) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Returns (order, mapping). `order` preserves category order of appearance.
    Any non-empty, non-URL line starts a new category. URL lines are assigned to
    the current category. If the file contains only URLs, mapping will be empty.
    """
    order: List[str] = []
    mapping: Dict[str, List[str]] = {}
    current: Optional[str] = None
    lines = path.read_text(encoding="utf-8").splitlines()
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        is_url = line.lower().startswith("http") or line.startswith("//")
        if is_url:
            if current is None:
                continue  # flat list mode, will be handled separately
            mapping.setdefault(current, []).append(_normalize_url(line))
        else:
            current = line
            if current not in mapping:
                mapping[current] = []
                order.append(current)
    # de-dup per category
    for k, urls in mapping.items():
        seen, dedup = set(), []
        for u in urls:
            if u and u not in seen:
                seen.add(u)
                dedup.append(u)
        mapping[k] = dedup
    return order, mapping


# ---------- your existing scrape_product ----------
def scrape_product(url: str, options: Options) -> Dict[str, Any]:
    driver = webdriver.Chrome(options=options)
    wait = WebDriverWait(driver, 20)
    try:
        driver.get(url)
        wait.until(lambda d: d.execute_script("return document.readyState") == "complete")

        try:
            wait.until(EC.presence_of_element_located((By.XPATH, "//script[@type='application/ld+json']")))
        except TimeoutException:
            pass

        # crude minimal extraction (you can paste your full logic here)
        product_schema = {}
        try:
            schema_el = driver.find_element(By.ID, "product-schema")
            product_schema = json.loads(schema_el.get_attribute("textContent"))
        except Exception:
            pass

        return {
            "url": url,
            "name": product_schema.get("name"),
            "sku": product_schema.get("sku"),
            "description": product_schema.get("description"),
        }
    finally:
        driver.quit()


# ---------- main ----------
def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape H&M product pages.")
    parser.add_argument("-i", "--input", help="Path to text file with URLs or categories+URLs")
    parser.add_argument("-o", "--output", help="Output JSON file (flat list mode)")
    parser.add_argument("--outdir", help="Directory for per-category JSON files")
    parser.add_argument("--pause", type=float, default=0.0, help="Seconds between requests")
    parser.add_argument("--jitter", type=float, default=0.0, help="Extra random seconds (0..jitter)")
    args = parser.parse_args()

    # Selenium options
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--lang=tr-TR")

    # input handling
    if args.input:
        p = Path(args.input)
        order, mapping = parse_links_with_categories(p)

        # categorized mode
        if mapping and order:
            outdir = Path(args.outdir or "out_json")
            outdir.mkdir(parents=True, exist_ok=True)
            for cat in order:
                urls = mapping.get(cat, [])
                results: List[Dict[str, Any]] = []
                print(f"\n=== {cat} ({len(urls)} URLs) ===")
                for idx, u in enumerate(urls, 1):
                    try:
                        print(f"[{idx}/{len(urls)}] {u}")
                        results.append(scrape_product(u, options))
                    except Exception as exc:
                        print(f"[WARN] {u}: {exc}")
                    # pause
                    if args.pause or args.jitter:
                        t = args.pause + (random.uniform(0, args.jitter) if args.jitter else 0.0)
                        time.sleep(t)
                safe_name = re.sub(r"[^A-Za-z0-9]+", "-", cat).strip("-").lower() or "uncategorized"
                out_path = outdir / f"{safe_name}.json"
                with open(out_path, "w", encoding="utf-8") as fh:
                    json.dump(results, fh, ensure_ascii=False, indent=2)
                print(f"[OK] {out_path}")
            return

        # flat list mode
        urls = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not args.output:
            print("[ERROR] please provide -o for flat list mode")
            return
        results = []
        for idx, u in enumerate(urls, 1):
            try:
                print(f"[{idx}/{len(urls)}] {u}")
                results.append(scrape_product(u, options))
            except Exception as exc:
                print(f"[WARN] {u}: {exc}")
            if args.pause or args.jitter:
                t = args.pause + (random.uniform(0, args.jitter) if args.jitter else 0.0)
                time.sleep(t)
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(results, fh, ensure_ascii=False, indent=2)
        print(f"[OK] {args.output}")
        return

    # interactive single url
    single_url = input("Enter H&M URL: ").strip()
    if not single_url:
        return
    data = [scrape_product(single_url, options)]
    out_path = Path(args.output or "output.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    print(f"[OK] {out_path}")


if __name__ == "__main__":
    main()