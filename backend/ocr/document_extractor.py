import os
import re
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger("healthai.ocr")

# Lazy-loaded EasyOCR reader instance
_easyocr_reader = None

def get_ocr_reader():
    global _easyocr_reader
    if _easyocr_reader is None:
        try:
            import easyocr
            # CPU mode with no verbose bar to prevent Windows charmap encoding issues
            _easyocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR: {e}")
            _easyocr_reader = False
    return _easyocr_reader if _easyocr_reader is not False else None


# Known clinical parameters with canonical names, regex aliases, standard units, and baseline reference thresholds
CLINICAL_PARAM_CATALOG = {
    # 1. Complete Blood Count (CBC) / Hemogram
    "hemoglobin": {
        "name": "Hemoglobin (Hb)",
        "aliases": [r"\bhemoglobin\b", r"\bhb\b", r"\bhgb\b"],
        "unit": "g/dL",
        "default_low": 12.0, "default_high": 17.5
    },
    "wbc": {
        "name": "Total Leucocyte Count (WBC)",
        "aliases": [r"\bwbc\b", r"\btotal leucocyte\b", r"\btlc\b", r"\bleukocytes?\b", r"\bwhite blood cells?\b"],
        "unit": "/cumm",
        "default_low": 4000.0, "default_high": 11000.0
    },
    "platelets": {
        "name": "Platelet Count",
        "aliases": [r"\bplatelets?\b", r"\bplt\b", r"\bplatelet count\b"],
        "unit": "thou/uL",
        "default_low": 150.0, "default_high": 450.0
    },
    "rbc": {
        "name": "Red Blood Cell Count (RBC)",
        "aliases": [r"\brbc\b", r"\bred blood cell\b"],
        "unit": "mil/uL",
        "default_low": 4.0, "default_high": 5.9
    },
    "mcv": {
        "name": "Mean Corpuscular Volume (MCV)",
        "aliases": [r"\bmcv\b"],
        "unit": "fL",
        "default_low": 80.0, "default_high": 100.0
    },
    "mch": {
        "name": "Mean Corpuscular Hemoglobin (MCH)",
        "aliases": [r"\bmch\b"],
        "unit": "pg",
        "default_low": 27.0, "default_high": 33.0
    },
    "mchc": {
        "name": "MCHC",
        "aliases": [r"\bmchc\b"],
        "unit": "g/dL",
        "default_low": 32.0, "default_high": 36.0
    },
    "esr": {
        "name": "Erythrocyte Sedimentation Rate (ESR)",
        "aliases": [r"\besr\b"],
        "unit": "mm/hr",
        "default_low": 0.0, "default_high": 20.0
    },
    "hematocrit": {
        "name": "Hematocrit (PCV)",
        "aliases": [r"\bhematocrit\b", r"\bpcv\b"],
        "unit": "%",
        "default_low": 36.0, "default_high": 50.0
    },

    # 2. Thyroid Profile
    "tsh": {
        "name": "Thyroid Stimulating Hormone (TSH)",
        "aliases": [r"\btsh\b", r"\bthyroid stimulating\b", r"\bultra-sensitive tsh\b"],
        "unit": "uIU/mL",
        "default_low": 0.40, "default_high": 4.50
    },
    "t3": {
        "name": "Total Triiodothyronine (T3)",
        "aliases": [r"\bt3\b", r"\btotal t3\b", r"\btriiodothyronine\b"],
        "unit": "ng/mL",
        "default_low": 0.80, "default_high": 2.00
    },
    "t4": {
        "name": "Total Thyroxine (T4)",
        "aliases": [r"\bt4\b", r"\btotal t4\b", r"\bthyroxine\b"],
        "unit": "ug/dL",
        "default_low": 5.0, "default_high": 12.0
    },

    # 3. Glycemic / Metabolic Panel
    "glucose_fasting": {
        "name": "Fasting Blood Glucose",
        "aliases": [r"\bfasting glucose\b", r"\bfasting blood sugar\b", r"\bfbs\b", r"\bglucose - fasting\b"],
        "unit": "mg/dL",
        "default_low": 70.0, "default_high": 99.0
    },
    "glucose_random": {
        "name": "Random / Postprandial Glucose",
        "aliases": [r"\brbs\b", r"\bppbs\b", r"\brbs\s*/\s*hgt\b", r"\brandom glucose\b", r"\bpostprandial\b"],
        "unit": "mg/dL",
        "default_low": 70.0, "default_high": 140.0
    },
    "hba1c": {
        "name": "Hemoglobin A1c (HbA1c)",
        "aliases": [r"\bhba1c\b", r"\bhb a1c\b", r"\bglycated hemoglobin\b"],
        "unit": "%",
        "default_low": 4.0, "default_high": 5.6
    },

    # 4. Lipid Profile
    "cholesterol_total": {
        "name": "Total Cholesterol",
        "aliases": [r"\btotal cholesterol\b", r"\bserum cholesterol\b"],
        "unit": "mg/dL",
        "default_low": 125.0, "default_high": 200.0
    },
    "cholesterol_hdl": {
        "name": "HDL Cholesterol",
        "aliases": [r"\bhdl cholesterol\b", r"\bhdl\b", r"\bhdl-c\b"],
        "unit": "mg/dL",
        "default_low": 40.0, "default_high": 60.0
    },
    "cholesterol_ldl": {
        "name": "LDL Cholesterol",
        "aliases": [r"\bldl cholesterol\b", r"\bldl\b", r"\bldl-c\b"],
        "unit": "mg/dL",
        "default_low": 50.0, "default_high": 100.0
    },
    "triglycerides": {
        "name": "Triglycerides",
        "aliases": [r"\btriglycerides?\b", r"\btg\b"],
        "unit": "mg/dL",
        "default_low": 50.0, "default_high": 150.0
    },

    # 5. Renal / Kidney & Electrolytes
    "creatinine": {
        "name": "Serum Creatinine",
        "aliases": [r"\bcreatinine\b", r"\bs\.?\s*creatinine\b", r"\bserum cr\b"],
        "unit": "mg/dL",
        "default_low": 0.6, "default_high": 1.2
    },
    "bun": {
        "name": "Blood Urea Nitrogen (BUN)",
        "aliases": [r"\bbun\b", r"\bblood urea nitrogen\b", r"\burea\b"],
        "unit": "mg/dL",
        "default_low": 7.0, "default_high": 20.0
    },
    "calcium": {
        "name": "Serum Calcium",
        "aliases": [r"\bcalcium\b", r"\bserum calcium\b"],
        "unit": "mg/dL",
        "default_low": 8.5, "default_high": 10.5
    },
    "phosphorus": {
        "name": "Serum Phosphorus",
        "aliases": [r"\bphosphorus\b", r"\bserum phosphorus\b"],
        "unit": "mg/dL",
        "default_low": 2.5, "default_high": 4.5
    },
    "sodium": {
        "name": "Serum Sodium",
        "aliases": [r"\bserum na\b", r"\bsodium\b"],
        "unit": "mEq/L",
        "default_low": 135.0, "default_high": 145.0
    },
    "potassium": {
        "name": "Serum Potassium",
        "aliases": [r"\bserum k\b", r"\bpotassium\b"],
        "unit": "mEq/L",
        "default_low": 3.5, "default_high": 5.0
    },

    # 6. Liver Function & Enzymes
    "sgot": {
        "name": "SGOT (AST)",
        "aliases": [r"\bsgot\b", r"\bast\b", r"\baspartate aminotransferase\b"],
        "unit": "U/L",
        "default_low": 10.0, "default_high": 40.0
    },
    "sgpt": {
        "name": "SGPT (ALT)",
        "aliases": [r"\bsgpt\b", r"\balt\b", r"\balanine aminotransferase\b"],
        "unit": "U/L",
        "default_low": 7.0, "default_high": 56.0
    },
    "bilirubin_total": {
        "name": "Total Bilirubin",
        "aliases": [r"\bt\.?\s*bill?\b", r"\btotal bilirubin\b"],
        "unit": "mg/dL",
        "default_low": 0.2, "default_high": 1.2
    },
    "albumin": {
        "name": "Serum Albumin",
        "aliases": [r"\balbumin\b", r"\bs\.?\s*albumin\b"],
        "unit": "g/dL",
        "default_low": 3.5, "default_high": 5.5
    },
    # 7. Vital Signs / Blood Pressure
    "systolic_bp": {
        "name": "Systolic Blood Pressure",
        "aliases": [r"\bsystolic\b", r"\bsbp\b", r"\bsystolic bp\b"],
        "unit": "mmHg",
        "default_low": 90.0, "default_high": 120.0
    },
    "diastolic_bp": {
        "name": "Diastolic Blood Pressure",
        "aliases": [r"\bdiastolic\b", r"\bdbp\b", r"\bdiastolic bp\b"],
        "unit": "mmHg",
        "default_low": 60.0, "default_high": 80.0
    }
}


import numpy as np

def preprocess_image_for_ocr(image_input):
    """
    Applies comprehensive image preprocessing for scanned, photographed,
    or low-contrast medical documents and prescriptions:
    1. Channel normalization (RGBA/BGR to Grayscale)
    2. Resolution upscaling (bicubic interpolation if < 1400px)
    3. Contrast enhancement (CLAHE)
    4. Edge-preserving noise reduction (Bilateral filter)
    5. Automatic skew detection & deskew rotation
    6. Adaptive Gaussian thresholding
    Returns: (enhanced_gray, thresholded_binary)
    """
    try:
        import cv2
        if isinstance(image_input, str):
            img = cv2.imread(image_input)
        else:
            img = image_input

        if img is None:
            return None, None

        # 1. Channel normalization & Upscaling if needed
        if len(img.shape) == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        h, w = img.shape[:2]
        if max(h, w) < 1400:
            scale = 1400.0 / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

        # 2. Grayscale conversion
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # 3. Contrast Limited Adaptive Histogram Equalization (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # 4. Bilateral filter: smooths noise while keeping text boundaries sharp
        filtered = cv2.bilateralFilter(enhanced, 9, 60, 60)

        # 5. Deskew correction
        try:
            dark_pixels = np.column_stack(np.where(filtered < 190))
            if len(dark_pixels) > 200:
                rect = cv2.minAreaRect(dark_pixels)
                angle = rect[-1]
                if angle < -45:
                    angle = -(90 + angle)
                elif angle > 45:
                    angle = 90 - angle
                else:
                    angle = -angle
                if 1.0 < abs(angle) < 40.0:
                    (hc, wc) = filtered.shape[:2]
                    M = cv2.getRotationMatrix2D((wc // 2, hc // 2), angle, 1.0)
                    filtered = cv2.warpAffine(filtered, M, (wc, hc), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                    enhanced = cv2.warpAffine(enhanced, M, (wc, hc), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        except Exception as deskew_err:
            logger.debug(f"Deskew bypassed: {deskew_err}")

        # 6. Adaptive Gaussian thresholding
        thresh = cv2.adaptiveThreshold(
            filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10
        )

        return filtered, thresh
    except Exception as e:
        logger.error(f"Image preprocessing error: {e}")
        return None, None


def ocr_image_advanced(file_path: str) -> Dict[str, Any]:
    """
    Executes multi-pass OCR with layout ordering and confidence scoring.
    Pass 1: CLAHE-enhanced grayscale (best for EasyOCR neural network).
    Pass 2: Adaptive binary thresholding if Pass 1 yields low word count.
    Returns: {"text": str, "confidence": float, "is_uncertain": bool, "word_count": int}
    """
    reader = get_ocr_reader()
    if not reader:
        return {"text": "", "confidence": 0.0, "is_uncertain": True, "word_count": 0}

    enhanced, thresh = preprocess_image_for_ocr(file_path)
    img_to_run = enhanced if enhanced is not None else file_path

    # Pass 1: Enhanced grayscale
    try:
        raw_results = reader.readtext(img_to_run, detail=1)
    except Exception as e:
        logger.error(f"EasyOCR Pass 1 failed on {file_path}: {e}")
        raw_results = []

    # Pass 2: Fall back to adaptive threshold if Pass 1 found very few elements
    if (len(raw_results) < 5) and (thresh is not None):
        try:
            raw_results_p2 = reader.readtext(thresh, detail=1)
            if len(raw_results_p2) > len(raw_results):
                raw_results = raw_results_p2
        except Exception as e:
            logger.error(f"EasyOCR Pass 2 failed on {file_path}: {e}")

    if not raw_results:
        return {"text": "", "confidence": 0.0, "is_uncertain": True, "word_count": 0}

    # Sort boxes in top-to-bottom, left-to-right reading order
    # Each box: [[tl_x, tl_y], [tr_x, tr_y], [br_x, br_y], [bl_x, bl_y]]
    def sort_key(item):
        bbox = item[0]
        y_top = min(pt[1] for pt in bbox)
        x_left = min(pt[0] for pt in bbox)
        # Quantize y into line buckets (e.g. 15px) to group words on the same line
        return (int(y_top // 15), x_left)

    sorted_results = sorted(raw_results, key=sort_key)

    lines = []
    current_bucket = None
    current_line_words = []
    confs = []

    for item in sorted_results:
        bbox, text, conf = item
        text_str = str(text).strip()
        if not text_str:
            continue
        confs.append(float(conf))
        y_top = min(pt[1] for pt in bbox)
        bucket = int(y_top // 15)

        if current_bucket is None:
            current_bucket = bucket
            current_line_words.append(text_str)
        elif abs(bucket - current_bucket) <= 1:
            current_line_words.append(text_str)
        else:
            lines.append(" ".join(current_line_words))
            current_line_words = [text_str]
            current_bucket = bucket

    if current_line_words:
        lines.append(" ".join(current_line_words))

    full_text = "\n".join(lines)
    avg_conf = (sum(confs) / len(confs)) if confs else 0.0
    is_uncertain = avg_conf < 0.55 or len(lines) < 2

    return {
        "text": full_text,
        "confidence": round(avg_conf, 2),
        "is_uncertain": is_uncertain,
        "word_count": len(confs)
    }


def extract_document_text(file_path: str) -> str:
    """
    Extracts text from an uploaded document (PDF or image).
    For PDFs: First attempts digital text extraction; falls back to OCR if scanned.
    For Images: Runs advanced preprocessed OCR.
    """
    ext = os.path.splitext(file_path)[1].lower()
    full_text = []

    if ext == ".pdf":
        try:
            import fitz
            doc = fitz.open(file_path)
            for i in range(len(doc)):
                page = doc[i]
                page_text = page.get_text().strip()
                if len(page_text) > 40:
                    full_text.append(page_text)
                else:
                    # Scanned PDF page: render to image and apply advanced OCR
                    pix = page.get_pixmap(dpi=150)
                    temp_img = f"{file_path}_p{i}.png"
                    pix.save(temp_img)
                    try:
                        res = ocr_image_advanced(temp_img)
                        if res.get("text"):
                            full_text.append(res["text"])
                    finally:
                        if os.path.exists(temp_img):
                            try:
                                os.remove(temp_img)
                            except Exception:
                                pass
            doc.close()
        except Exception as e:
            logger.error(f"Error extracting PDF text from {file_path}: {e}")

    elif ext in [".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp"]:
        res = ocr_image_advanced(file_path)
        if res.get("text"):
            full_text.append(res["text"])

    return "\n".join(full_text)


def detect_report_date(text: str) -> Optional[str]:
    """
    Searches document text for explicit medical report dates.
    Returns normalized 'YYYY-MM-DD' string, or None if uncertain.
    """
    lines = text.splitlines()

    # Pattern 1: Explicit 'Date: ...'
    date_prefix_pattern = re.compile(r"(?i)(?:date(?:\s+of\s+collection)?|dt|dated|test\s+date|coll(?:ected)?\s+date)[\s:=-]+([0-9]{2,4}[-/.][0-9]{1,2}[-/.][0-9]{2,4})")
    for line in lines:
        m = date_prefix_pattern.search(line)
        if m:
            d_str = m.group(1).replace('/', '-').replace('.', '-')
            parsed = parse_date_string(d_str)
            if parsed:
                return parsed

    # Pattern 2: DD/MM/YYYY or YYYY-MM-DD standalone near top
    date_pattern = re.compile(r"\b([0-9]{2,4})[-/.]([0-9]{1,2})[-/.]([0-9]{2,4})\b")
    for line in lines[:20]:
        matches = date_pattern.findall(line)
        for d, m, y in matches:
            if len(d) == 4:
                parsed = parse_date_parts(y, m, d)
            else:
                parsed = parse_date_parts(d, m, y)
            if parsed:
                return parsed

    # Pattern 3: Words like '18 Jan 2025' or 'January 18, 2025'
    month_names = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*"
    word_date = re.compile(rf"(?i)\b([0-9]{{1,2}})\s+({month_names})\s+([0-9]{{4}})\b")
    for line in lines[:20]:
        m = word_date.search(line)
        if m:
            try:
                dt = datetime.strptime(f"{m.group(1)} {m.group(2)[:3]} {m.group(3)}", "%d %b %Y")
                return dt.strftime("%Y-%m-%d")
            except Exception:
                pass

    return None


def parse_date_parts(d_str: str, m_str: str, y_str: str) -> Optional[str]:
    try:
        d = int(d_str)
        m = int(m_str)
        y = int(y_str)
        if y < 100:
            y += 2000
        if 2000 <= y <= 2035 and 1 <= m <= 12 and 1 <= d <= 31:
            return f"{y:04d}-{m:02d}-{d:02d}"
    except Exception:
        pass
    return None


def parse_date_string(s: str) -> Optional[str]:
    parts = s.split('-')
    if len(parts) == 3:
        if len(parts[0]) == 4:
            return parse_date_parts(parts[2], parts[1], parts[0])
        else:
            return parse_date_parts(parts[0], parts[1], parts[2])
    return None


def detect_lab_and_title(text: str) -> Tuple[str, str]:
    """
    Attempts to identify the laboratory/hospital name and report title from the document headers.
    """
    lab_name = "Clinical Diagnostic Laboratory"
    report_title = "Laboratory Medical Report"

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    header_chunk = " ".join(lines[:10]).lower()

    if "zrs hospital" in header_chunk:
        lab_name = "ZRS Hospital & Diagnostic Center"
    elif "apex" in header_chunk:
        lab_name = "Apex Clinical Laboratories"
    elif "metropolis" in header_chunk:
        lab_name = "Metropolis Healthcare"
    elif "dr. lal" in header_chunk or "pathlabs" in header_chunk:
        lab_name = "Dr. Lal PathLabs"
    else:
        for ln in lines[:5]:
            if any(k in ln.lower() for k in ["hospital", "clinic", "laborator", "pathology", "diagnostics"]):
                lab_name = ln[:60].strip()
                break

    if "investigation chart" in header_chunk:
        report_title = "Investigation Chart"
    elif "thyroid" in header_chunk:
        report_title = "Thyroid Function Panel"
    elif "complete blood" in header_chunk or "hemogram" in header_chunk or "cbc" in header_chunk:
        report_title = "Complete Blood Count (CBC)"
    elif "lipid" in header_chunk:
        report_title = "Lipid Profile Panel"
    elif "metabolic" in header_chunk:
        report_title = "Comprehensive Metabolic Panel"

    return lab_name, report_title


def extract_reference_range_from_line(line: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Extracts high and low reference values from text if present in the document.
    Examples: '12.0 - 16.0', '(70.0 - 99.0)', '< 200', '> 40', 'Ref: 4.0 - 5.6'
    """
    # Pattern: X.X - Y.Y
    range_match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*[-–—to]\s*([0-9]+(?:\.[0-9]+)?)", line)
    if range_match:
        try:
            low = float(range_match.group(1))
            high = float(range_match.group(2))
            if low <= high:
                return low, high
        except Exception:
            pass

    # Pattern: < Y.Y (Upper bound only)
    less_match = re.search(r"<\s*([0-9]+(?:\.[0-9]+)?)", line)
    if less_match:
        try:
            return None, float(less_match.group(1))
        except Exception:
            pass

    # Pattern: > X.X (Lower bound only)
    greater_match = re.search(r">\s*([0-9]+(?:\.[0-9]+)?)", line)
    if greater_match:
        try:
            return float(greater_match.group(1)), None
        except Exception:
            pass

    return None, None


def parse_laboratory_parameters(text: str) -> List[Dict[str, Any]]:
    """
    Extracts structured laboratory test parameters from text/OCR.
    Preserves all parameters present in the report without forcing a fixed schema.
    Extracts actual document reference ranges when present, otherwise utilizes clinical reference limits.
    Never invents missing parameters or values.
    """
    results = []
    lines = text.splitlines()
    found_keys = set()

    for key, meta in CLINICAL_PARAM_CATALOG.items():
        matched_val = None
        doc_low = None
        doc_high = None

        # Check line by line
        for i, line in enumerate(lines):
            line_clean = line.strip()
            if not line_clean:
                continue

            for alias in meta["aliases"]:
                if re.search(alias, line_clean, re.IGNORECASE):
                    # Check for numeric value in this line
                    # Discard matches that are parts of dates like 2025 or 18/06
                    num_matches = re.findall(r"\b([0-9]+(?:\.[0-9]+)?)\b", line_clean)
                    valid_nums = []
                    for n in num_matches:
                        try:
                            f = float(n)
                            # Exclude obvious 4-digit years like 2023, 2024, 2025 unless WBC or Platelet
                            if f in [2022, 2023, 2024, 2025, 2026] and key not in ["wbc", "platelets"]:
                                continue
                            valid_nums.append(f)
                        except ValueError:
                            pass

                    if valid_nums:
                        # Value is typically the first number right after the test name
                        matched_val = valid_nums[0]
                        # Check line for reference range
                        d_low, d_high = extract_reference_range_from_line(line_clean)
                        if d_low is not None or d_high is not None:
                            doc_low, doc_high = d_low, d_high
                        break

                    # If no number on this line, check the immediate next line (common in multi-column OCR)
                    if i + 1 < len(lines):
                        next_line = lines[i + 1].strip()
                        next_nums = re.findall(r"\b([0-9]+(?:\.[0-9]+)?)\b", next_line)
                        if next_nums:
                            try:
                                f = float(next_nums[0])
                                if f not in [2022, 2023, 2024, 2025, 2026] or key in ["wbc", "platelets"]:
                                    matched_val = f
                                    break
                            except ValueError:
                                pass

            if matched_val is not None:
                break

        if matched_val is not None and key not in found_keys:
            found_keys.add(key)
            ref_low = doc_low if doc_low is not None else meta.get("default_low")
            ref_high = doc_high if doc_high is not None else meta.get("default_high")

            # Determine clinical status
            status = "NORMAL"
            if ref_high is not None and matched_val > ref_high:
                status = "HIGH"
            elif ref_low is not None and matched_val < ref_low:
                status = "LOW"

            results.append({
                "test_name": meta["name"],
                "result_value": round(matched_val, 2),
                "unit": meta["unit"],
                "reference_low": ref_low,
                "reference_high": ref_high,
                "status": status,
                "clinical_explanation": f"Observed {meta['name']} is {matched_val} {meta['unit']} (Standard: {ref_low or 0} - {ref_high or '—'} {meta['unit']})."
            })

    # Check for combined Blood Pressure (e.g., 120/80 mmHg or BP: 130/85)
    if "systolic_bp" not in found_keys or "diastolic_bp" not in found_keys:
        bp_match = re.search(r"(?:b\.?p\.?|blood\s*pressure)?[:\s]*([0-9]{2,3})\s*/\s*([0-9]{2,3})(?:\s*mm\s*hg)?", text, re.IGNORECASE)
        if bp_match:
            try:
                sys_val = float(bp_match.group(1))
                dia_val = float(bp_match.group(2))
                if 60 <= sys_val <= 250 and 40 <= dia_val <= 150:
                    if "systolic_bp" not in found_keys:
                        found_keys.add("systolic_bp")
                        results.append({
                            "test_name": "Systolic Blood Pressure",
                            "result_value": sys_val,
                            "unit": "mmHg",
                            "reference_low": 90.0,
                            "reference_high": 120.0,
                            "status": "NORMAL" if sys_val <= 120 else "HIGH",
                            "clinical_explanation": f"Observed Systolic BP is {sys_val} mmHg."
                        })
                    if "diastolic_bp" not in found_keys:
                        found_keys.add("diastolic_bp")
                        results.append({
                            "test_name": "Diastolic Blood Pressure",
                            "result_value": dia_val,
                            "unit": "mmHg",
                            "reference_low": 60.0,
                            "reference_high": 80.0,
                            "status": "NORMAL" if dia_val <= 80 else "HIGH",
                            "clinical_explanation": f"Observed Diastolic BP is {dia_val} mmHg."
                        })
            except (ValueError, TypeError):
                pass

    return results


def extract_report_metadata(text: str) -> Dict[str, Any]:
    """
    Extracts patient age, gender, or patient name from document header text if explicitly printed.
    Never invents values.
    """
    meta = {}
    age_match = re.search(r"\b(?:age|age\s*/\s*sex|years\s*old)[:\s]*([0-9]{1,3})\s*(?:yrs|years|y)?\b", text, re.IGNORECASE)
    if age_match:
        try:
            val = int(age_match.group(1))
            if 1 <= val <= 120:
                meta["age"] = val
        except ValueError:
            pass

    gender_match = re.search(r"\b(?:gender|sex|age\s*/\s*sex)[:\s]*(?:[0-9]{1,3}\s*(?:yrs|years|y)?\s*[/,-]?\s*)?(male|female|m\b|f\b)", text, re.IGNORECASE)
    if gender_match:
        g = gender_match.group(1).lower()
        if g in ["m", "male"]:
            meta["gender"] = "Male"
        elif g in ["f", "female"]:
            meta["gender"] = "Female"

    return meta
