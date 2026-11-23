import re
from typing import List, Dict, Any, Optional

MEDICINE_DATABASE = {
    "metformin": "Metformin is a biguanide antidiabetic medication commonly prescribed to help lower blood glucose levels and improve insulin sensitivity in individuals with Type 2 diabetes.",
    "lisinopril": "Lisinopril is an ACE (angiotensin-converting enzyme) inhibitor commonly prescribed to treat high blood pressure (hypertension) and protect kidney and cardiovascular health.",
    "atorvastatin": "Atorvastatin is a statin medication widely prescribed to lower low-density lipoprotein (LDL or 'bad') cholesterol and triglycerides, reducing cardiovascular risk.",
    "amlodipine": "Amlodipine is a calcium channel blocker used to treat high blood pressure and prevent chest pain (angina) by relaxing blood vessels.",
    "donepezil": "Donepezil is a cholinesterase inhibitor used to support cognitive function, memory, and daily tasks in patients experiencing Alzheimer's disease symptoms.",
    "aspirin": "Low-dose aspirin is commonly prescribed as an antiplatelet agent to help prevent blood clots in individuals with elevated cardiovascular or stroke risk.",
    "losartan": "Losartan is an angiotensin II receptor blocker (ARB) used to treat high blood pressure and protect kidney function in diabetic patients.",
    "telmisartan": "Telmisartan is an angiotensin II receptor blocker (ARB) used to manage high blood pressure (hypertension) and reduce cardiovascular risk.",
    "metoprolol": "Metoprolol is a beta-blocker that affects the heart and circulation, used to treat angina and hypertension.",
    "pantoprazole": "Pantoprazole is a proton pump inhibitor that decreases stomach acid, used for GERD and acid reflux.",
    "glimepiride": "Glimepiride is a sulfonylurea antidiabetic medication that stimulates insulin secretion to manage blood sugar.",
    "levothyroxine": "Levothyroxine is a synthetic thyroid hormone used to treat hypothyroidism (underactive thyroid gland).",
    "omeprazole": "Omeprazole is a proton pump inhibitor (PPI) prescribed to reduce stomach acid and treat gastroesophageal reflux disease (GERD) or gastritis.",
    "paracetamol": "Paracetamol (acetaminophen) is an analgesic and antipyretic used to relieve mild-to-moderate pain and reduce fever.",
    "ibuprofen": "Ibuprofen is a nonsteroidal anti-inflammatory drug (NSAID) used to relieve pain, swelling, and fever.",
    "amoxicillin": "Amoxicillin is a penicillin antibiotic used to treat bacterial infections such as respiratory, ear, or throat infections.",
    "azithromycin": "Azithromycin is a macrolide antibiotic used to treat various bacterial infections.",
    "ciprofloxacin": "Ciprofloxacin is a fluoroquinolone antibiotic used to treat serious bacterial infections.",
    "cetirizine": "Cetirizine is an antihistamine that reduces the effects of natural chemical histamine in allergic conditions.",
    "clopidogrel": "Clopidogrel is an antiplatelet medication used to prevent blood clots in coronary artery disease or stroke.",
    "furosemide": "Furosemide is a loop diuretic used to treat fluid build-up (edema) due to heart failure, liver, or kidney disease.",
    "gabapentin": "Gabapentin is an anticonvulsant and neuropathic pain medication used for nerve pain and seizures."
}

from .document_extractor import parse_date_string, extract_document_text, ocr_image_advanced

def parse_prescription_text(text: str, ocr_conf: Optional[float] = None) -> Dict[str, Any]:
    """
    Extracts doctor, date, diagnosis, follow-up, tests advised, and structured medicines
    from prescription text. Flags uncertainty when confidence is low or fields are ambiguous.
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    doctor_name = "Unknown Doctor"
    prescription_date = ""
    diagnosis = ""
    follow_up = ""
    tests_advised = ""
    medicines = []
    seen_med_names = set()

    # 1. Look for Doctor Name
    doc_patterns = [
        r"(?i)(?:dr\.|doctor|physician|consultant)[\s:]+([A-Za-z\s.,'-]+?)(?:\s*(?:md|mbbs|ms|facp|phd|do|,|\n|$))",
        r"(?i)^(dr\.\s+[A-Za-z\s.,'-]+)"
    ]
    for line in lines[:12]:
        for pat in doc_patterns:
            m = re.search(pat, line)
            if m and doctor_name == "Unknown Doctor":
                extracted_doc = m.group(1).strip() if m.lastindex else m.group(0).strip()
                # Clean up extracted doc name
                cleaned = re.sub(r'[\r\n]+', '', extracted_doc).strip()
                if len(cleaned) > 3 and not re.match(r'(?i)^(doctor|physician)$', cleaned):
                    if not cleaned.lower().startswith("dr.") and not cleaned.lower().startswith("dr "):
                        cleaned = "Dr. " + cleaned
                    doctor_name = cleaned
                break
        if doctor_name != "Unknown Doctor":
            break

    # 2. Look for Prescription Date
    date_pattern = re.compile(r"\b([0-9]{2,4})[-/.]([0-9]{1,2})[-/.]([0-9]{2,4})\b")
    for line in lines[:15]:
        m = date_pattern.search(line)
        if m and not prescription_date:
            parsed = parse_date_string(m.group(0))
            if parsed:
                prescription_date = parsed
                break

    if not prescription_date:
        for line in lines:
            m = date_pattern.search(line)
            if m:
                parsed = parse_date_string(m.group(0))
                if parsed:
                    prescription_date = parsed
                    break

    # 3. Look for Diagnosis, Tests, Follow-up
    for i, line in enumerate(lines):
        lower_line = line.lower()
        if ("diagnosis:" in lower_line or "dx:" in lower_line or "assessment:" in lower_line) and not diagnosis:
            diagnosis = re.sub(r'(?i)^(diagnosis|dx|assessment)[\s:]*', '', line).strip()
            if not diagnosis and i + 1 < len(lines):
                diagnosis = lines[i+1].strip()
        if ("follow-up" in lower_line or "follow up" in lower_line or "review after" in lower_line or "next visit" in lower_line) and not follow_up:
            follow_up = re.sub(r'(?i)^(follow-up|follow up|review after|next visit)[\s:]*', '', line).strip()
            if not follow_up and i + 1 < len(lines):
                follow_up = lines[i+1].strip()
        if ("tests advised" in lower_line or "investigations" in lower_line or "lab tests" in lower_line or "advise:" in lower_line) and not tests_advised:
            tests_advised = re.sub(r'(?i)^(tests advised|investigations|lab tests|advise)[\s:]*', '', line).strip()
            if not tests_advised and i + 1 < len(lines):
                tests_advised = lines[i+1].strip()

    # 4. Parse Medicines: Known Catalog
    for line in lines:
        for med_key, purpose in MEDICINE_DATABASE.items():
            if re.search(rf"\b{med_key}\b", line, re.IGNORECASE) and med_key not in seen_med_names:
                seen_med_names.add(med_key)
                # Extract dosage
                dose_match = re.search(r"([0-9]+\s*(?:mg|mcg|g|ml|iu))", line, re.IGNORECASE)
                dosage = dose_match.group(1) if dose_match else "As directed"

                # Extract frequency & timing
                freq = "Once daily"
                timing = "Morning"
                if re.search(r"twice|bid|b\.i\.d|2 times|1-0-1", line, re.IGNORECASE):
                    freq = "Twice daily"
                    timing = "Morning, Night"
                elif re.search(r"thrice|tid|t\.i\.d|3 times|1-1-1", line, re.IGNORECASE):
                    freq = "Three times daily"
                    timing = "Morning, Afternoon, Night"
                elif re.search(r"bedtime|night|hs|h\.s|0-0-1", line, re.IGNORECASE):
                    freq = "Once daily"
                    timing = "Night"
                elif re.search(r"1-0-0|morning", line, re.IGNORECASE):
                    freq = "Once daily"
                    timing = "Morning"

                # Meal instruction
                inst = "Take with a full glass of water after food."
                meal = "After meal"
                if re.search(r"empty stomach|before meals?|before food|a\.?c\.?", line, re.IGNORECASE):
                    inst = "Take on an empty stomach before meals."
                    meal = "Before meal"

                # Duration
                dur_match = re.search(r"([0-9]+\s*(?:days?|weeks?|months?))", line, re.IGNORECASE)
                duration = dur_match.group(1) if dur_match else "30 days"

                medicines.append({
                    "medicine_name": med_key.capitalize(),
                    "dosage": dosage,
                    "frequency": freq,
                    "timing": timing,
                    "meal_instruction": meal,
                    "duration": duration,
                    "instructions": inst,
                    "purpose_explanation": purpose
                })
                break

    # 5. Extract Candidate Medicines from lines with Tab/Cap/Dosage (for prescriptions with unlisted/handwritten items)
    if not medicines:
        cand_pattern = re.compile(r"(?i)\b(?:tab|cap|syr|inj)?\.?\s*([A-Za-z]{3,20})\s+([0-9]+\s*(?:mg|mcg|g|ml))\b")
        for line in lines:
            m = cand_pattern.search(line)
            if m:
                c_name = m.group(1).capitalize()
                c_dose = m.group(2)
                if c_name.lower() not in seen_med_names and len(c_name) > 3:
                    seen_med_names.add(c_name.lower())
                    medicines.append({
                        "medicine_name": c_name,
                        "dosage": c_dose,
                        "frequency": "As directed",
                        "timing": "Morning",
                        "meal_instruction": "After meal",
                        "duration": "Duration per prescription",
                        "instructions": "Take as directed by your physician.",
                        "purpose_explanation": f"Medication prescribed for clinical condition. Please verify indications with your physician."
                    })

    # Confidence calculation & uncertainty evaluation
    conf_val = ocr_conf if ocr_conf is not None else (0.85 if medicines else 0.40)
    is_uncertain = (conf_val < 0.60) or (len(medicines) == 0) or (doctor_name == "Unknown Doctor" and not prescription_date)

    review_notes = ""
    if is_uncertain:
        if len(medicines) == 0:
            review_notes = "No standard medications automatically recognized. Please enter or review prescribed medicines."
        elif conf_val < 0.60:
            review_notes = "Handwritten or low-contrast document detected. Please review extracted doctor and medication details before confirming."

    return {
        "doctor_name": doctor_name,
        "prescription_date": prescription_date,
        "diagnosis": diagnosis,
        "follow_up": follow_up,
        "tests_advised": tests_advised,
        "medicines": medicines,
        "raw_text": text,
        "confidence": round(conf_val, 2),
        "is_uncertain": is_uncertain,
        "needs_review": is_uncertain,
        "review_notes": review_notes
    }


def extract_prescription_from_file(file_path: str, mime_type: str) -> Dict[str, Any]:
    """
    Extracts text using advanced image preprocessing and multi-pass OCR for images,
    or digital/OCR extraction for PDFs.
    """
    ext = (file_path.split(".")[-1] if "." in file_path else "").lower()
    
    ocr_confidence = None
    if ext in ["png", "jpg", "jpeg", "webp", "tiff", "bmp"]:
        res = ocr_image_advanced(file_path)
        text = res.get("text", "")
        ocr_confidence = res.get("confidence", 0.0)
    else:
        text = extract_document_text(file_path)

    if not text:
        return {
            "doctor_name": "Unknown Doctor",
            "prescription_date": "",
            "diagnosis": "",
            "follow_up": "",
            "tests_advised": "",
            "medicines": [],
            "raw_text": "",
            "confidence": 0.0,
            "is_uncertain": True,
            "needs_review": True,
            "review_notes": "Could not read text from document. Please verify the document is legible or enter details manually."
        }

    return parse_prescription_text(text, ocr_conf=ocr_confidence)
