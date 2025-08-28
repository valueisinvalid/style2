import json
import re
from typing import Any, Dict, List, Optional

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import argparse


TARGET_URL = "https://www2.hm.com/tr_tr/productpage.1309903002.html"


def read_script_json_by_id(driver: webdriver.Chrome, script_id: str, wait: WebDriverWait) -> Dict[str, Any]:
    """Wait for a <script> by id, read its textContent, and parse JSON.

    Falls back to parsing the page source directly if the element cannot be located in time.
    """
    try:
        script_element = wait.until(EC.presence_of_element_located((By.ID, script_id)))
        raw_text = script_element.get_attribute("textContent") or script_element.get_attribute("innerText")
        if not raw_text or not raw_text.strip():
            raise ValueError(f"Script tag with id '{script_id}' is empty or missing text content")
        return json.loads(raw_text)
    except TimeoutException:
        # Fallback 1: direct DOM JS access (handles shadow DOM/serialization quirks)
        raw_text = driver.execute_script(
            """
            var el = document.getElementById(arguments[0]);
            return el ? (el.textContent || el.innerText) : null;
            """,
            script_id,
        )
        if raw_text and str(raw_text).strip():
            return json.loads(raw_text)
        # Fallback 2: parse the page source manually
        page = driver.page_source
        pattern = rf"<script[^>]*id=\"{re.escape(script_id)}\"[^>]*>(.*?)</script>"
        match = re.search(pattern, page, flags=re.DOTALL | re.IGNORECASE)
        if match:
            raw_text = match.group(1)
            return json.loads(raw_text)
        raise


def read_first_jsonld_of_type(driver: webdriver.Chrome, wait: WebDriverWait, type_name: str) -> Optional[Dict[str, Any]]:
    """Scan all <script type="application/ld+json"> tags and return the first object matching @type.

    Handles the case where the script contains a JSON array of objects.
    """
    try:
        wait.until(EC.presence_of_element_located((By.XPATH, "//script[@type='application/ld+json']")))
    except TimeoutException:
        # Continue; we may still parse from page_source
        pass

    scripts = driver.find_elements(By.XPATH, "//script[@type='application/ld+json']")
    candidates: List[str] = []
    for s in scripts:
        text = s.get_attribute("textContent") or s.get_attribute("innerText")
        if text and text.strip():
            candidates.append(text)
    if not candidates:
        # Try from page source as a last resort
        page = driver.page_source
        for m in re.finditer(r"<script[^>]*type=\"application/ld\+json\"[^>]*>([\s\S]*?)</script>", page, flags=re.IGNORECASE):
            candidates.append(m.group(1))

    for raw in candidates:
        try:
            data = json.loads(raw)
        except Exception:
            continue
        # Sometimes this is a list of JSON-LD objects
        if isinstance(data, list):
            for obj in data:
                if isinstance(obj, dict) and (obj.get("@type") == type_name or (isinstance(obj.get("@type"), list) and type_name in obj.get("@type"))):
                    return obj
        elif isinstance(data, dict):
            if data.get("@type") == type_name or (isinstance(data.get("@type"), list) and type_name in data.get("@type")):
                return data
    return None


def safe_get(dct: Any, path: List[Any], default: Any = None) -> Any:
    """Safely navigate a nested dict/list structure using a path list."""
    current = dct
    for key in path:
        try:
            if isinstance(key, int) and isinstance(current, list):
                current = current[key]
            elif isinstance(current, dict):
                current = current.get(key)
            else:
                return default
        except (IndexError, KeyError, TypeError):
            return default
        if current is None:
            return default
    return current


def deep_find_first_key(node: Any, target_key: str) -> Optional[Any]:
    """Depth-first search for the first value by key name in nested dict/list structures."""
    if isinstance(node, dict):
        if target_key in node:
            return node.get(target_key)
        for value in node.values():
            found = deep_find_first_key(value, target_key)
            if found is not None:
                return found
    elif isinstance(node, list):
        for item in node:
            found = deep_find_first_key(item, target_key)
            if found is not None:
                return found
    return None


def normalise_url(url: Optional[str]) -> Optional[str]:
    """Ensure protocol-relative URLs start with https:// and return a clean string."""
    if not url:
        return None
    s = str(url).strip()
    if not s:
        return None
    if s.startswith("//"):
        return "https:" + s
    return s


def extract_product_image(
    driver: webdriver.Chrome,
    wait: WebDriverWait,
    product_schema: Dict[str, Any],
    next_data_json: Dict[str, Any],
) -> Optional[str]:
    """Extract the **still-life (product-only)** image URL using H&M's own metadata.

    Priority (most reliable to least):
    0) `__NEXT_DATA__` → first image whose `assetType` is exactly ``DESCRIPTIVESTILLLIFE``.
       (Located under props.pageProps.productData.articlesList[...].images or anywhere in the JSON.)
    1) product-schema JSON-LD: `image` (string or first element of list)
    2) `<meta property="og:image">`
    3) Other `__NEXT_DATA__` keys: `ogImage`, `thumbnailImg`, `imageUrl`, or any asset list
    4) Fallback: first <img> element found in the gallery / page
    """

    # 0) __NEXT_DATA__ – look for DESCRIPTIVESTILLLIFE images explicitly
    def find_still_life_in_images(images: Any) -> Optional[str]:
        """Return baseUrl or image where assetType == DESCRIPTIVESTILLLIFE."""
        if not isinstance(images, list):
            return None
        for img_info in images:
            if not isinstance(img_info, dict):
                continue
            if img_info.get("assetType") == "DESCRIPTIVESTILLLIFE":
                return normalise_url(img_info.get("baseUrl") or img_info.get("image"))
        return None

    # Explicit productData → articlesList path (fast path)
    articles = safe_get(
        next_data_json,
        ["props", "pageProps", "productData", "articlesList"],
    )
    if isinstance(articles, list):
        for art in articles:
            still = find_still_life_in_images(art.get("images") if isinstance(art, dict) else None)
            if still:
                return still

    # Deep search fallback across entire JSON (handles future structure changes)
    def deep_search_for_still(node: Any) -> Optional[str]:
        if isinstance(node, dict):
            # If this dict itself has the keys we're interested in, check immediately
            if "assetType" in node and node.get("assetType") == "DESCRIPTIVESTILLLIFE":
                return normalise_url(node.get("baseUrl") or node.get("image"))
            # Otherwise recurse
            for v in node.values():
                found = deep_search_for_still(v)
                if found:
                    return found
        elif isinstance(node, list):
            for item in node:
                found = deep_search_for_still(item)
                if found:
                    return found
        return None

    still_life = deep_search_for_still(next_data_json)
    if still_life:
        return still_life

    # 1) JSON-LD Product "image"
    image_field = product_schema.get("image")
    if isinstance(image_field, str) and image_field.strip():
        img = normalise_url(image_field)
        if img:
            return img
    if isinstance(image_field, list) and image_field:
        for candidate in image_field:
            img = normalise_url(candidate)
            if img:
                return img

    # 2) Open Graph image
    try:
        og = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "meta[property='og:image']")))
        og_content = og.get_attribute("content")
        img = normalise_url(og_content)
        if img:
            return img
    except TimeoutException:
        pass

    # 3) __NEXT_DATA__ deep keys
    for key in ("ogImage", "thumbnailImg", "imageUrl"):
        val = deep_find_first_key(next_data_json, key)
        img = normalise_url(val) if isinstance(val, str) else None
        if img:
            return img
    # assets list with objects containing baseUrl or image
    assets = deep_find_first_key(next_data_json, "assets")
    if isinstance(assets, list):
        for asset in assets:
            if not isinstance(asset, dict):
                continue
            for k in ("baseUrl", "image"):
                img = normalise_url(asset.get(k))
                if img:
                    return img

    # 4) Fallback: any image inside a gallery container
    try:
        # Common containers include data-testid="swipe-gallery" or role="img"
        gallery = driver.find_elements(By.CSS_SELECTOR, "[data-testid='swipe-gallery'] img, .product-detail-main-image-container img, img")
        for el in gallery:
            src = el.get_attribute("src") or el.get_attribute("data-src")
            img = normalise_url(src)
            if img and "image.hm.com" in img:
                return img
        # Last-ditch: first img anywhere
        if gallery:
            src = gallery[0].get_attribute("src") or gallery[0].get_attribute("data-src")
            return normalise_url(src)
    except Exception:
        pass

    return None


def format_fabric_composition(next_data_json: Dict[str, Any]) -> Optional[str]:
    """Extract fabric composition from Next.js data.

    The structure observed in H&M's __NEXT_DATA__ can vary.  Two patterns are
    common:

    1.  "compositions": ["Polyester 100%", "Elastane 5%", ...]  # list[str]
    2.  "composition":   [{"materials": [{"name": "Polyester", "amount": "100%"}, ...]}]

    The original implementation only handled (2).  This revision supports both
    formats and also searches for the singular key "composition" when the
    plural is not present.
    """

    # 1. Collect the first non-empty candidate list from the most specific to
    #    the most generic locations.
    candidates: List[Any] = []
    for key in ("compositions", "composition"):
        val = safe_get(next_data_json, ["props", "pageProps", "productData", key])
        if val:
            candidates.append(val)
            break  # Prefer the explicit productData path

    if not candidates:
        # Deep-search fallbacks
        for key in ("compositions", "composition"):
            val = deep_find_first_key(next_data_json, key)
            if val:
                candidates.append(val)
                break

    if not candidates:
        return None

    compositions = candidates[0]

    # Handle list[str] – already in final format, just normalise & join
    if isinstance(compositions, list) and compositions and isinstance(compositions[0], str):
        cleaned = [s.strip() for s in compositions if isinstance(s, str) and s.strip()]
        return ", ".join(cleaned) if cleaned else None

    # Handle list[dict] with nested materials
    if isinstance(compositions, list) and compositions and isinstance(compositions[0], dict):
        materials: List[Dict[str, Any]] = []
        for comp in compositions:
            if not isinstance(comp, dict):
                continue
            mats = comp.get("materials")
            if isinstance(mats, list):
                materials.extend(mats)

        parts: List[str] = []
        for material in materials:
            if not isinstance(material, dict):
                continue
            name = (material.get("name") or "").strip()
            amount_raw = material.get("amount")
            amount = str(amount_raw).strip() if amount_raw is not None else ""

            # Normalise amount – append % when numeric without a percent sign
            if amount and not amount.endswith("%"):
                try:
                    float(amount.replace(",", "."))
                    amount += "%"
                except ValueError:
                    pass

            piece = f"{name} {amount}".strip()
            if piece:
                parts.append(piece)

        return ", ".join(parts) if parts else None

    # Unknown structure – give up gracefully
    return None


def extract_key_value_description(next_data_json: Dict[str, Any]) -> Dict[str, str]:
    attributes = safe_get(next_data_json, ["props", "pageProps", "productData", "attributes"], [])
    if not attributes:
        attributes = deep_find_first_key(next_data_json, "attributes") or []
    key_value: Dict[str, str] = {}
    if not isinstance(attributes, list):
        return key_value
    for attr in attributes:
        if not isinstance(attr, dict):
            continue
        title = attr.get("title")
        values = attr.get("values")
        if not title:
            continue
        if isinstance(values, list):
            value_str = ", ".join([v for v in values if isinstance(v, str) and v.strip()])
        elif values is None:
            value_str = ""
        else:
            value_str = str(values)
        key_value[title] = value_str
    return key_value


def find_label_for_key(next_data_json: Dict[str, Any], key_code: str) -> Optional[str]:
    """Find a localized label for a given attribute code by scanning dicts where the key exists with a string value."""
    stack: List[Any] = [next_data_json]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            if key_code in node and isinstance(node[key_code], str):
                return node[key_code]
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    return None


def build_attributes_from_variant(next_data_json: Dict[str, Any], variant: Dict[str, Any]) -> Dict[str, str]:
    product_attributes = safe_get(variant, ["productAttributes", "description"], [])
    kv: Dict[str, str] = {}
    if not isinstance(product_attributes, list):
        return kv
    for item in product_attributes:
        if not isinstance(item, dict):
            continue
        code = item.get("title")
        values = item.get("values")
        if not code:
            continue
        label = find_label_for_key(next_data_json, code) or code
        if isinstance(values, list):
            # Join measurement-like lists with spaces; color lists with comma + space for readability
            joiner = ", " if code == "detailedDescriptions" else " "
            value_str = joiner.join([v for v in values if isinstance(v, str) and v.strip()])
        elif isinstance(values, str):
            value_str = values
        else:
            value_str = ""
        if label and value_str is not None:
            # Ensure label ends with colon like in examples
            label_out = label if label.endswith(":") else f"{label}:"
            kv[label_out] = value_str
    return kv


def scrape_product(url: str, options: Options) -> Dict[str, Any]:
    """Scrape a single H&M product page and return the parsed JSON payload.

    This is a refactor of the scraping logic that previously lived directly in
    ``main`` so that it can be re-used in batch mode (e.g. when the user passes
    a urls.txt file).  The implementation is identical to the original logic
    with the following differences:

    1. It does NOT prompt for any user input or write individual ``*.json``
       files – it simply returns the parsed dictionary.
    2. The Chrome ``webdriver`` is created and torn down *inside* this helper so
       that each product page is isolated (simpler and avoids state bleed).
    """
    driver = webdriver.Chrome(options=options)
    wait = WebDriverWait(driver, 20)
    try:
        # Navigate to the product URL and wait for the DOM to be ready.
        driver.get(url)
        wait.until(lambda d: d.execute_script("return document.readyState") == "complete")

        # Ensure at least one JSON-LD block is present (helps with hydration timing)
        try:
            wait.until(EC.presence_of_element_located((By.XPATH, "//script[@type='application/ld+json']")))
        except TimeoutException:
            pass  # Continue – we'll fall back to page-source parsing if needed

        # 1) Primary product schema JSON-LD (id="product-schema")
        try:
            product_schema = read_script_json_by_id(driver, "product-schema", wait)
        except Exception:
            product_schema = read_first_jsonld_of_type(driver, wait, "Product") or {}

        item_id: Optional[str] = product_schema.get("sku")
        name = product_schema.get("name")
        gender = safe_get(product_schema, ["audience", "suggestedGender"])
        description = product_schema.get("description")

        # 2) Breadcrumb schema – we use the second-to-last element to infer the product type
        try:
            breadcrumb_schema = read_script_json_by_id(driver, "breadcrumb-schema", wait)
        except Exception:
            breadcrumb_schema = read_first_jsonld_of_type(driver, wait, "BreadcrumbList") or {}
        item_list = breadcrumb_schema.get("itemListElement") or []
        product_type = None
        if isinstance(item_list, list) and len(item_list) >= 2:
            second_last = item_list[-2]
            if isinstance(second_last, dict):
                product_type = second_last.get("name")

        # 3) __NEXT_DATA__ – this is where H&M stores most dynamic product info
        try:
            next_data_json = read_script_json_by_id(driver, "__NEXT_DATA__", wait)
        except Exception:
            page_src = driver.page_source
            m = re.search(r"<script[^>]*id=\"__NEXT_DATA__\"[^>]*>([\s\S]*?)</script>", page_src, flags=re.IGNORECASE)
            if m:
                next_data_json = json.loads(m.group(1))
            else:
                # Fallback – scan any <script> blocks looking for Next.js payloads
                next_data_json = {}
                for s in driver.find_elements(By.TAG_NAME, "script"):
                    txt = s.get_attribute("textContent") or ""
                    if txt.strip().startswith("{") and "pageProps" in txt and "productData" in txt:
                        try:
                            cand = json.loads(txt)
                        except Exception:
                            continue
                        if safe_get(cand, ["props", "pageProps", "productData"], None) is not None:
                            next_data_json = cand
                            break

        # Variant-specific data – colour, attributes, still-life image, etc.
        color = None
        key_value_description: Dict[str, str] = {}
        variant_still: Optional[str] = None
        current_sku = item_id
        variations_map = deep_find_first_key(next_data_json, "variations")
        if isinstance(variations_map, dict) and current_sku and current_sku in variations_map:
            # Recursively locate the variant dict whose swatchDetails.sku == current_sku
            def _find_variant(node: Any, sku: str) -> Optional[Dict[str, Any]]:
                if isinstance(node, dict):
                    sd = node.get("swatchDetails")
                    if isinstance(sd, dict) and sd.get("sku") == sku:
                        return node
                    for v in node.values():
                        res = _find_variant(v, sku)
                        if res is not None:
                            return res
                elif isinstance(node, list):
                    for item in node:
                        res = _find_variant(item, sku)
                        if res is not None:
                            return res
                return None

            current_variant = _find_variant(next_data_json, current_sku)
            if isinstance(current_variant, dict):
                color = safe_get(current_variant, ["swatchDetails", "colorName"]) or current_variant.get("name")
                key_value_description = build_attributes_from_variant(next_data_json, current_variant)

                def _variant_still_life(variant_dict: Optional[Dict[str, Any]]) -> Optional[str]:
                    if not isinstance(variant_dict, dict):
                        return None
                    images = variant_dict.get("images")
                    if not isinstance(images, list):
                        return None
                    for img in images:
                        if isinstance(img, dict) and img.get("assetType") == "DESCRIPTIVESTILLLIFE":
                            return normalise_url(img.get("baseUrl") or img.get("image"))
                    return None

                variant_still = _variant_still_life(current_variant)

        if not color:
            articles = deep_find_first_key(next_data_json, "articlesList")
            if isinstance(articles, list) and articles:
                first_article = articles[0]
                if isinstance(first_article, dict):
                    color = safe_get(first_article, ["color", "name"]) or first_article.get("colorName")
        if not key_value_description:
            key_value_description = extract_key_value_description(next_data_json)

        fabric_composition = format_fabric_composition(next_data_json)
        product_image = variant_still or extract_product_image(driver, wait, product_schema, next_data_json)

        result: Dict[str, Any] = {
            "item_id": item_id,
            "name": name,
            "type": product_type,
            "color": color,
            "gender": gender,
            "fabric_composition": fabric_composition,
            "description": description,
            "key_value_description": key_value_description,
            "productImage": product_image,
        }

        if not item_id:
            raise ValueError("Failed to extract a valid 'item_id' from the product page; skipping.")

        return result
    finally:
        driver.quit()


def main() -> None:
    """Entry-point that supports both single-URL and batch scraping modes."""
    parser = argparse.ArgumentParser(description="Scrape H&M product pages and output JSON.")
    parser.add_argument("-i", "--input", help="Path to a text file with one URL per line.")
    parser.add_argument("-o", "--output", required=True, help="Path of the aggregated JSON output file.")
    args = parser.parse_args()

    # Re-use the same Chrome Options configuration from the original code
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--lang=tr-TR")
    options.add_argument("--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36")

    # Determine which URLs to scrape
    urls: List[str] = []
    if args.input:
        try:
            with open(args.input, "r", encoding="utf-8") as fh:
                urls = [ln.strip() for ln in fh if ln.strip()]
        except Exception as exc:
            print(f"[ERROR] Failed to read input file '{args.input}': {exc}")
            return
    else:
        single_url = input("Enter the H&M product URL to scrape: ").strip()
        if not single_url:
            print("No URL provided. Exiting.")
            return
        urls = [single_url]

    aggregated: List[Dict[str, Any]] = []
    for u in urls:
        try:
            print(f"Scraping: {u}")
            aggregated.append(scrape_product(u, options))
        except Exception as exc:
            print(f"[WARN] Skipping '{u}' – {exc}")

    if not aggregated:
        print("No data scraped. Exiting without creating an output file.")
        return

    try:
        with open(args.output, "w", encoding="utf-8") as out_fh:
            json.dump(aggregated, out_fh, ensure_ascii=False, indent=2)
        print(f"Aggregated data for {len(aggregated)} product(s) saved to '{args.output}'.")
    except Exception as exc:
        print(f"[ERROR] Could not write output file '{args.output}': {exc}")


if __name__ == "__main__":
    main()


