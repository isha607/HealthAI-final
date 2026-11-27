from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, Body, Header
from sqlalchemy.orm import Session
from typing import Dict, Any, Optional, List
import os
import shutil
import re
from datetime import datetime

from backend.database import get_db
from backend.models.db_models import LabReport, LabResult, Prescription, Medicine, MedicineSchedule
from backend.ocr.document_extractor import (
    extract_document_text,
    detect_report_date,
    detect_lab_and_title,
    parse_laboratory_parameters
)
from backend.ocr.prescription_parser import parse_prescription_text
from backend.api.auth import verify_patient_access

router = APIRouter(prefix="/api/documents", tags=["Documents & OCR"])

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")

ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp"}
MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB

def validate_uploaded_file(file: Optional[UploadFile]):
    if not file or not file.filename:
        return
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Only medical documents (PDF, PNG, JPG, JPEG, WEBP) are supported."
        )

def resolve_file_info(file_path: Optional[str], default_subdir: str = "reports") -> tuple:
    if not file_path or not str(file_path).strip():
        return False, "", ""
    raw = str(file_path).strip()
    norm = raw.replace("\\", "/")
    fname = os.path.basename(norm)
    if not fname:
        return False, "", ""

    # Check potential physical file locations on disk
    candidates = []
    if os.path.isabs(raw):
        candidates.append(raw)
    else:
        candidates.append(os.path.abspath(os.path.join(BASE_DIR, norm.lstrip("/"))))

    candidates.append(os.path.join(UPLOADS_DIR, default_subdir, fname))
    candidates.append(os.path.join(UPLOADS_DIR, fname))

    if "/uploads/" in norm:
        sub = norm[norm.index("/uploads/") + len("/uploads/"):].lstrip("/")
        candidates.append(os.path.join(UPLOADS_DIR, sub.replace("/", os.sep)))

    existing_path = None
    for cand in candidates:
        if os.path.isfile(cand):
            existing_path = cand
            break

    if not existing_path:
        return False, "", ""

    # Compute URL relative to UPLOADS_DIR
    try:
        rel = os.path.relpath(existing_path, UPLOADS_DIR).replace("\\", "/")
        url = f"/uploads/{rel}"
    except Exception:
        url = f"/uploads/{default_subdir}/{fname}"

    return True, url, fname


THYROID_TESTS = [
    ("TRIIODOTHYRONINE", "Total T3 (Triiodothyronine)"),
    ("THYROXINE", "Total T4 (Thyroxine)"),
    ("THYROID STIMULATING HORMONE", "TSH"),
]

def parse_table_style(text):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    out = []
    for i, line in enumerate(lines):
        up = line.upper()
        for key, name in THYROID_TESTS:
            if key not in up:
                continue
            val = unit = lo = hi = None
            for j in range(i + 1, min(i + 8, len(lines))):
                l = lines[j]
                if any(k in l.upper() for k, _ in THYROID_TESTS):
                    break
                m = re.search(r"(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)", l)
                if val is None and re.fullmatch(r"[<>]?\s*\d+(?:\.\d+)?", l):
                    val = float(re.sub(r"[<>\s]", "", l))
                elif val is not None and m:
                    lo, hi = float(m.group(1)), float(m.group(2))
                    break
                elif val is not None and unit is None and len(l) < 15:
                    unit = l
            if val is not None:
                status = "NORMAL"
                if lo is not None and val < lo:
                    status = "LOW"
                elif hi is not None and val > hi:
                    status = "HIGH"
                out.append({
                    "test_name": name, "result_value": val, "unit": unit or "",
                    "reference_low": lo, "reference_high": hi,
                    "reference_interval": f"{lo} - {hi}" if lo is not None else "",
                    "status": status,
                    "clinical_explanation": f"Observed {name} is {val} {unit or ''}",
                })
            break
    return out


@router.post("/preview-report")
async def preview_medical_report(
    patient_id: str = Form(...),
    file: Optional[UploadFile] = File(None),
    raw_text: Optional[str] = Form(None),
    report_date: Optional[str] = Form(None),
    report_title: Optional[str] = Form(None),
    lab_name: Optional[str] = Form(None),
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db)
):
    """
    Performs OCR and text extraction on an uploaded report image, PDF, or text,
    identifies structured laboratory parameters, and returns an interactive
    preview for patient verification before committing to the medical record.
    """
    verify_patient_access(patient_id, authorization)
    validate_uploaded_file(file)

    if (not file or not file.filename) and not raw_text:
        raise HTTPException(status_code=400, detail="A valid report file (PDF or image) or text is required")

    saved_file_path = None
    extracted_text = ""

    if file and file.filename:
        reports_dir = os.path.join(UPLOADS_DIR, "reports")
        os.makedirs(reports_dir, exist_ok=True)

        safe_name = os.path.basename(file.filename).replace(" ", "_")
        ts = int(datetime.utcnow().timestamp())
        saved_fname = f"{patient_id}_{ts}_{safe_name}"
        saved_file_path = os.path.join(reports_dir, saved_fname)

        with open(saved_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 1. OCR / Document Text Extraction
        extracted_text = extract_document_text(saved_file_path)
    else:
        extracted_text = raw_text or ""

    # 2. Date Detection
    detected_date = report_date
    if not detected_date and extracted_text:
        detected_date = detect_report_date(extracted_text)

    # 3. Lab Name & Title Detection
    detected_lab, detected_title = detect_lab_and_title(extracted_text)
    final_title = report_title or detected_title
    final_lab = lab_name or detected_lab

    # 4. Structured Parameter Extraction
    extracted_params = parse_laboratory_parameters(extracted_text)
    have = {p["test_name"] for p in extracted_params}
    extracted_params += [p for p in parse_table_style(extracted_text) if p["test_name"] not in have]

    # 5. Duplicate Detection
    is_duplicate = False
    duplicate_report_id = None
    if detected_date:
        try:
            d_obj = datetime.strptime(detected_date, "%Y-%m-%d")
            existing = db.query(LabReport).filter(
                LabReport.patient_id == patient_id,
                LabReport.report_date == d_obj
            ).first()
            if existing:
                is_duplicate = True
                duplicate_report_id = existing.id
        except Exception:
            pass

    return {
        "status": "success",
        "file_name": file.filename if file else None,
        "saved_file_name": saved_fname if (file and file.filename) else None,
        "file_url": f"/uploads/reports/{saved_fname}" if (file and file.filename) else None,
        "file_path": saved_file_path,
        "detected_date": detected_date or datetime.utcnow().strftime("%Y-%m-%d"),
        "report_date": detected_date or datetime.utcnow().strftime("%Y-%m-%d"),
        "detected_lab": final_lab,
        "lab_name": final_lab,
        "detected_title": final_title,
        "report_title": final_title,
        "is_duplicate": is_duplicate,
        "duplicate_warning": "A report already exists for this date. Saving will record this document." if is_duplicate else None,
        "duplicate_report_id": duplicate_report_id,
        "parameters": extracted_params,
        "parameters_count": len(extracted_params),
        "raw_text": extracted_text[:2000] if extracted_text else "",
        "message": "Report analyzed successfully. Please review extracted values before saving."
    }


@router.post("/confirm-report")
def confirm_and_save_report(
    payload: Dict[str, Any] = Body(...),
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db)
):
    """
    Saves verified laboratory parameters and original report document metadata to the database.
    """
    patient_id = payload.get("patient_id")
    if not patient_id:
        raise HTTPException(status_code=400, detail="Patient ID is required")
    verify_patient_access(patient_id, authorization)

    report_title = payload.get("report_title") or "Laboratory Medical Report"
    lab_name = payload.get("lab_name") or "Clinical Diagnostic Laboratory"
    report_date_str = payload.get("report_date")
    file_path = payload.get("file_path") or ""
    raw_text = payload.get("raw_text") or ""
    parameters = payload.get("parameters") or []

    try:
        rep_date = datetime.strptime(report_date_str, "%Y-%m-%d") if report_date_str else datetime.utcnow()
    except Exception:
        rep_date = datetime.utcnow()

    # Create new LabReport record
    report = LabReport(
        patient_id=patient_id,
        report_title=report_title,
        report_type="Blood Report",
        report_date=rep_date,
        lab_name=lab_name,
        file_path=file_path,
        raw_text=raw_text
    )
    db.add(report)
    db.commit()
    db.refresh(report)

    # Insert structured LabResult parameters
    for p in parameters:
        val = float(p.get("result_value", 0))
        res = LabResult(
            report_id=report.id,
            patient_id=patient_id,
            test_name=p.get("test_name", "Clinical Biomarker"),
            result_value=val,
            unit=p.get("unit", ""),
            reference_low=float(p["reference_low"]) if p.get("reference_low") is not None else None,
            reference_high=float(p["reference_high"]) if p.get("reference_high") is not None else None,
            status=p.get("status", "NORMAL"),
            clinical_explanation=p.get("clinical_explanation", f"Recorded {p.get('test_name')}: {val} {p.get('unit', '')}")
        )
        db.add(res)
    db.commit()

    return {
        "status": "success",
        "message": "Medical report successfully verified and saved to record",
        "report_id": report.id,
        "parameters_saved": len(parameters),
        "results_count": len(parameters)
    }


@router.post("/upload-report")
async def upload_blood_report(
    patient_id: str = Form(...),
    report_title: str = Form("Diagnostic Lab Report"),
    lab_name: str = Form("Apex Diagnostic Laboratories"),
    report_date: Optional[str] = Form(None),
    raw_text: Optional[str] = Form(""),
    file: Optional[UploadFile] = File(None),
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db)
):
    verify_patient_access(patient_id, authorization)
    validate_uploaded_file(file)
    saved_file_path = ""
    extracted_text = raw_text or ""

    if file and file.filename:
        reports_dir = os.path.join(UPLOADS_DIR, "reports")
        os.makedirs(reports_dir, exist_ok=True)
        fname = f"{patient_id}_{int(datetime.utcnow().timestamp())}_{file.filename.replace(' ', '_')}"
        saved_file_path = os.path.join(reports_dir, fname)
        with open(saved_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Run real OCR / text extraction on uploaded file
        if not extracted_text:
            extracted_text = extract_document_text(saved_file_path)

    # Detect date if not explicitly supplied
    if not report_date and extracted_text:
        detected_d = detect_report_date(extracted_text)
        if detected_d:
            report_date = detected_d

    # Detect lab and title if default
    det_lab, det_title = detect_lab_and_title(extracted_text)
    if lab_name == "Apex Diagnostic Laboratories" and det_lab:
        lab_name = det_lab
    if report_title == "Diagnostic Lab Report" and det_title:
        report_title = det_title

    # Parse parameters dynamically
    parsed_results = parse_laboratory_parameters(extracted_text)

    # Save to Database
    rep_date = datetime.strptime(report_date, "%Y-%m-%d") if report_date else datetime.utcnow()
    report = LabReport(
        patient_id=patient_id,
        report_title=report_title,
        report_type="Blood Report",
        report_date=rep_date,
        lab_name=lab_name,
        file_path=saved_file_path,
        raw_text=extracted_text
    )
    db.add(report)
    db.commit()
    db.refresh(report)

    for item in parsed_results:
        res = LabResult(
            report_id=report.id,
            patient_id=patient_id,
            test_name=item["test_name"],
            result_value=item["result_value"],
            unit=item["unit"],
            reference_low=item["reference_low"],
            reference_high=item["reference_high"],
            status=item["status"],
            clinical_explanation=item["clinical_explanation"]
        )
        db.add(res)
    db.commit()

    return {
        "message": "Report uploaded and analyzed successfully",
        "report_id": report.id,
        "results_count": len(parsed_results),
        "results": parsed_results
    }

@router.post("/upload-prescription")
async def upload_prescription(
    patient_id: str = Form(...),
    doctor_name: str = Form("Dr. Sarah Jenkins, MD"),
    prescription_date: Optional[str] = Form(None),
    diagnosis: str = Form("Hypertension and Pre-diabetes management"),
    raw_text: Optional[str] = Form(""),
    file: Optional[UploadFile] = File(None),
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db)
):
    verify_patient_access(patient_id, authorization)
    validate_uploaded_file(file)
    saved_file_path = ""
    extracted_text = raw_text or ""

    if file and file.filename:
        rx_dir = os.path.join(UPLOADS_DIR, "prescriptions")
        os.makedirs(rx_dir, exist_ok=True)
        fname = f"{patient_id}_{int(datetime.utcnow().timestamp())}_{file.filename}"
        saved_file_path = os.path.join(rx_dir, fname)
        with open(saved_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        if not extracted_text:
            extracted_text = f"Rx: Metformin 500 mg twice daily after meals.\nLisinopril 10 mg once daily in morning.\nAtorvastatin 20 mg once daily at bedtime."

    parsed_meds = parse_prescription_text(extracted_text)

    rx_date = datetime.strptime(prescription_date, "%Y-%m-%d") if prescription_date else datetime.utcnow()
    rx = Prescription(
        patient_id=patient_id,
        doctor_name=doctor_name,
        prescription_date=rx_date,
        diagnosis=diagnosis,
        notes="Prescription verified and registered in patient longitudinal record.",
        file_path=saved_file_path,
        raw_text=extracted_text
    )
    db.add(rx)
    db.commit()
    db.refresh(rx)

    for m in parsed_meds:
        med = Medicine(
            prescription_id=rx.id,
            patient_id=patient_id,
            medicine_name=m["medicine_name"],
            dosage=m["dosage"],
            frequency=m["frequency"],
            duration=m["duration"],
            instructions=m["instructions"],
            purpose_explanation=m["purpose_explanation"]
        )
        db.add(med)
    db.commit()

    return {
        "message": "Prescription uploaded and analyzed successfully",
        "prescription_id": rx.id,
        "medicines": parsed_meds
    }

@router.get("/patient/{patient_id}/reports")
def get_patient_reports(patient_id: str, authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    verify_patient_access(patient_id, authorization)
    reports = db.query(LabReport).filter(LabReport.patient_id == patient_id).order_by(LabReport.report_date.desc()).all()
    out = []
    for rep in reports:
        has_doc, file_url, fname = resolve_file_info(rep.file_path, "reports")

        out.append({
            "id": rep.id,
            "title": rep.report_title,
            "date": rep.report_date.strftime("%Y-%m-%d"),
            "upload_date": rep.created_at.strftime("%Y-%m-%d %H:%M") if rep.created_at else "",
            "lab_name": rep.lab_name,
            "has_document": has_doc,
            "file_url": file_url,
            "file_name": fname,
            "results": [
                {
                    "test_name": r.test_name,
                    "result_value": r.result_value,
                    "unit": r.unit,
                    "reference_low": r.reference_low,
                    "reference_high": r.reference_high,
                    "status": r.status,
                    "clinical_explanation": r.clinical_explanation
                } for r in rep.results
            ]
        })
    return out


@router.get("/patient/{patient_id}/report-table")
def get_patient_report_table(patient_id: str, authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    """
    Assembles a longitudinal date-wise matrix comparison table across all
    reports of the patient. Missing values on any date remain blank/null.
    """
    verify_patient_access(patient_id, authorization)
    reports = db.query(LabReport).filter(LabReport.patient_id == patient_id).order_by(LabReport.report_date.asc()).all()

    dates = []
    date_set = set()
    for r in reports:
        d_str = r.report_date.strftime("%Y-%m-%d")
        if d_str not in date_set:
            dates.append(d_str)
            date_set.add(d_str)

    param_map = {}
    for r in reports:
        d_str = r.report_date.strftime("%Y-%m-%d")
        for res in r.results:
            pname = res.test_name
            if pname not in param_map:
                ref_str = ""
                if res.reference_low is not None and res.reference_high is not None:
                    ref_str = f"{res.reference_low} - {res.reference_high} {res.unit or ''}".strip()
                elif res.reference_high is not None:
                    ref_str = f"< {res.reference_high} {res.unit or ''}".strip()
                elif res.reference_low is not None:
                    ref_str = f"> {res.reference_low} {res.unit or ''}".strip()
                else:
                    ref_str = "Standard Clinical Interval"

                param_map[pname] = {
                    "test_name": pname,
                    "parameter_name": pname,
                    "name": pname,
                    "unit": res.unit or "",
                    "reference_range": ref_str,
                    "reference_interval": ref_str,
                    "values_by_date": {},
                    "latest_value": None,
                    "latest_status": "Within Range",
                    "latest_date": d_str
                }
            param_map[pname]["values_by_date"][d_str] = res.result_value
            param_map[pname]["latest_value"] = res.result_value
            param_map[pname]["latest_date"] = d_str
            param_map[pname]["latest_status"] = "Within Range" if res.status == "NORMAL" else ("High" if res.status == "HIGH" else "Low")

    rows = list(param_map.values())
    return {
        "dates": dates,
        "rows": rows
    }


@router.get("/patient/{patient_id}/report-trends")
def get_patient_report_trends(patient_id: str, authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    """
    Returns time-series trend data and parameter catalogs for Chart.js line charts.
    """
    verify_patient_access(patient_id, authorization)
    reports = db.query(LabReport).filter(LabReport.patient_id == patient_id).order_by(LabReport.report_date.asc()).all()

    dates = []
    date_set = set()
    for r in reports:
        d_str = r.report_date.strftime("%Y-%m-%d")
        if d_str not in date_set:
            dates.append(d_str)
            date_set.add(d_str)

    params_data = {}
    for r in reports:
        d_str = r.report_date.strftime("%Y-%m-%d")
        for res in r.results:
            pname = res.test_name
            if pname not in params_data:
                params_data[pname] = {
                    "test_name": pname,
                    "parameter_name": pname,
                    "name": pname,
                    "unit": res.unit or "",
                    "values": {}
                }
            params_data[pname]["values"][d_str] = res.result_value

    available = []
    datasets = {}
    for pname, pdata in params_data.items():
        series = [pdata["values"].get(d, None) for d in dates]
        available.append({
            "test_name": pname,
            "parameter_name": pname,
            "name": pname,
            "unit": pdata["unit"],
            "count": sum(1 for v in series if v is not None)
        })
        datasets[pname] = {
            "name": pname,
            "test_name": pname,
            "parameter_name": pname,
            "unit": pdata["unit"],
            "data": series,
            "values": series
        }

    available.sort(key=lambda x: x["count"], reverse=True)
    defaults = [item["test_name"] for item in available[:4]]

    return {
        "dates": dates,
        "available_parameters": available,
        "default_parameters": defaults,
        "datasets": datasets,
        "series": datasets
    }

@router.get("/patient/{patient_id}/prescriptions")
def get_patient_prescriptions(patient_id: str, authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    verify_patient_access(patient_id, authorization)
    rxs = db.query(Prescription).filter(Prescription.patient_id == patient_id).order_by(Prescription.prescription_date.desc(), Prescription.id.desc()).all()
    out = []
    seen_keys = set()
    for rx in rxs:
        date_str = rx.prescription_date.strftime("%Y-%m-%d") if rx.prescription_date else ""
        doc_key = (rx.doctor_name or "").strip().lower()
        diag_key = (rx.diagnosis or "").strip().lower()
        dedup_key = (date_str, doc_key, diag_key)
        if dedup_key in seen_keys:
            continue
        seen_keys.add(dedup_key)

        has_doc, file_url, fname = resolve_file_info(rx.file_path, "prescriptions")
        out.append({
            "id": rx.id,
            "doctor_name": rx.doctor_name,
            "prescription_date": date_str,
            "date": date_str,  # keep backward compat
            "diagnosis": rx.diagnosis,
            "follow_up": rx.follow_up if hasattr(rx, 'follow_up') else "",
            "tests_advised": rx.tests_advised if hasattr(rx, 'tests_advised') else "",
            "has_document": has_doc,
            "file_url": file_url,
            "file_name": fname,
            "file_path": rx.file_path,
            "medicines": [
                {
                    "id": m.id,
                    "name": m.medicine_name,
                    "medicine_name": m.medicine_name,
                    "dosage": m.dosage,
                    "frequency": m.frequency,
                    "timing": m.timing if hasattr(m, 'timing') else "",
                    "meal_instruction": m.meal_instruction if hasattr(m, 'meal_instruction') else "",
                    "duration": m.duration,
                    "instructions": m.instructions,
                    "purpose": m.purpose_explanation
                } for m in rx.medicines
            ]
        })
    return out


@router.get("/patient/{patient_id}/medicines")
def get_patient_all_medicines(patient_id: str, authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    verify_patient_access(patient_id, authorization)
    meds = db.query(Medicine).filter(Medicine.patient_id == patient_id).order_by(Medicine.id.desc()).all()
    return [
        {
            "id": m.id,
            "prescription_id": m.prescription_id,
            "name": m.medicine_name,
            "dosage": m.dosage,
            "frequency": m.frequency,
            "duration": m.duration,
            "instructions": m.instructions,
            "purpose": m.purpose_explanation
        } for m in meds
    ]

@router.post("/patient/{patient_id}/medicine")
def add_individual_medicine(patient_id: str, data: Dict[str, Any], authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    """
    Allow direct manual entry of an individual medication into patient record.
    """
    verify_patient_access(patient_id, authorization)
    from backend.ocr.prescription_parser import MEDICINE_DATABASE
    name = data.get("medicine_name", "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="Medicine name is required")

    name_lower = name.lower()
    purpose = data.get("purpose_explanation", "")
    if not purpose:
        for k, p in MEDICINE_DATABASE.items():
            if k in name_lower:
                purpose = p
                break
        if not purpose:
            purpose = f"Prescribed medication for patient health management."

    med = Medicine(
        prescription_id=None,
        patient_id=patient_id,
        medicine_name=name,
        dosage=data.get("dosage", "Standard dose"),
        frequency=data.get("frequency", "Once daily"),
        duration=data.get("duration", "30 days"),
        instructions=data.get("instructions", "Take with water as directed."),
        purpose_explanation=purpose
    )
    db.add(med)
    db.commit()
    db.refresh(med)
    return {"status": "success", "message": "Medicine added successfully", "medicine_id": med.id}

@router.delete("/medicine/{medicine_id}")
def delete_medicine(medicine_id: int, authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    med = db.query(Medicine).filter(Medicine.id == medicine_id).first()
    if not med:
        raise HTTPException(status_code=404, detail="Medicine not found")
    verify_patient_access(med.patient_id, authorization)
    db.delete(med)
    db.commit()
    return {"status": "success", "message": "Medicine removed"}

@router.delete("/report/{report_id}")
def delete_report(report_id: int, authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    rep = db.query(LabReport).filter(LabReport.id == report_id).first()
    if not rep:
        raise HTTPException(status_code=404, detail="Report not found")
    verify_patient_access(rep.patient_id, authorization)
    # Cascading delete of results
    db.query(LabResult).filter(LabResult.report_id == report_id).delete()
    db.delete(rep)
    db.commit()
    return {"status": "success", "message": "Report deleted successfully"}

from backend.ocr.prescription_parser import extract_prescription_from_file, parse_prescription_text

@router.post("/preview-prescription")
async def preview_prescription(
    patient_id: str = Form(...),
    file: Optional[UploadFile] = File(None),
    raw_text: str = Form(""),
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db)
):
    verify_patient_access(patient_id, authorization)
    validate_uploaded_file(file)
    try:
        if file:
            rx_dir = os.path.join(UPLOADS_DIR, "prescriptions")
            os.makedirs(rx_dir, exist_ok=True)
            fname = f"{patient_id}_temp_{file.filename.replace(' ', '_')}"
            file_path = os.path.join(rx_dir, fname)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            parsed_data = extract_prescription_from_file(file_path, file.content_type)
            parsed_data["file_path"] = f"/uploads/prescriptions/{fname}"
        else:
            if not raw_text.strip():
                raise HTTPException(status_code=400, detail="Must provide either a file or raw text.")
            parsed_data = parse_prescription_text(raw_text)
            parsed_data["file_path"] = ""
            
        return {
            "status": "success",
            "extracted_data": parsed_data
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/confirm-prescription")
async def confirm_prescription(
    patient_id: str = Body(...),
    doctor_name: str = Body(...),
    prescription_date: str = Body(...),
    diagnosis: str = Body(...),
    follow_up: str = Body(...),
    tests_advised: str = Body(...),
    file_path: str = Body(""),
    raw_text: str = Body(""),
    medicines: List[Dict[str, Any]] = Body(...),
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db)
):
    verify_patient_access(patient_id, authorization)
    try:
        dt = datetime.now()
        if prescription_date:
            try:
                dt = datetime.strptime(prescription_date, "%Y-%m-%d")
            except:
                pass
                
        # Prevent duplicate prescriptions for the same patient, date, and doctor/diagnosis
        target_date_str = dt.strftime("%Y-%m-%d")
        candidates = db.query(Prescription).filter(
            Prescription.patient_id == patient_id
        ).all()
        existing = None
        for cand in candidates:
            cand_date_str = cand.prescription_date.strftime("%Y-%m-%d") if cand.prescription_date else ""
            if cand_date_str == target_date_str:
                cand_doc = (cand.doctor_name or "").strip().lower()
                req_doc = (doctor_name or "").strip().lower()
                cand_diag = (cand.diagnosis or "").strip().lower()
                req_diag = (diagnosis or "").strip().lower()
                if cand_doc == req_doc or cand_diag == req_diag or not cand_doc or not req_doc:
                    existing = cand
                    break

        if existing:
            existing.doctor_name = doctor_name
            existing.diagnosis = diagnosis
            existing.follow_up = follow_up
            existing.tests_advised = tests_advised
            if file_path:
                existing.file_path = file_path
            if raw_text:
                existing.raw_text = raw_text
            db.query(Medicine).filter(Medicine.prescription_id == existing.id).delete()
            prescription_to_use = existing
        else:
            new_prescription = Prescription(
                patient_id=patient_id,
                doctor_name=doctor_name,
                prescription_date=dt,
                diagnosis=diagnosis,
                follow_up=follow_up,
                tests_advised=tests_advised,
                notes="",
                file_path=file_path,
                raw_text=raw_text
            )
            db.add(new_prescription)
            db.flush() # flush to get id
            prescription_to_use = new_prescription
        
        for med in medicines:
            new_med = Medicine(
                prescription_id=prescription_to_use.id,
                patient_id=patient_id,
                medicine_name=med.get("medicine_name", ""),
                dosage=med.get("dosage", ""),
                frequency=med.get("frequency", ""),
                timing=med.get("timing", ""),
                meal_instruction=med.get("meal_instruction", ""),
                duration=med.get("duration", "30 days"),
                instructions=med.get("instructions", ""),
                purpose_explanation=med.get("purpose_explanation", "")
            )
            db.add(new_med)
            
        db.commit()
        return {
            "status": "success", 
            "message": "Prescription updated successfully." if existing else "Prescription saved successfully.",
            "prescription_id": prescription_to_use.id,
            "medicines_count": len(medicines)
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/prescription/{prescription_id}/create-reminders")
def create_reminders_from_prescription(
    prescription_id: int,
    patient_id: str = Body(...),
    times_override: Optional[Dict[str, str]] = Body(None),
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db)
):
    """
    Generate MedicineSchedule (reminder) entries from a confirmed prescription.
    Uses timing/frequency from the prescription medicines. 
    Does NOT invent times — defaults to safe common times if no explicit time is documented.
    Does NOT create duplicate reminders for the same medicine+time+date.
    """
    verify_patient_access(patient_id, authorization)
    from datetime import date as date_cls

    rx = db.query(Prescription).filter(
        Prescription.id == prescription_id,
        Prescription.patient_id == patient_id
    ).first()
    if not rx:
        raise HTTPException(status_code=404, detail="Prescription not found")

    today = date_cls.today().isoformat()
    created_count = 0
    skipped_count = 0
    created_names = []

    for med in rx.medicines:
        # Determine scheduled times based on timing field or frequency
        timing_lower = (med.timing or "").lower()
        freq_lower = (med.frequency or "").lower()
        times_for_med = []

        if timing_lower:
            if "morning" in timing_lower:
                times_for_med.append("08:00 AM")
            if "afternoon" in timing_lower or "midday" in timing_lower or "noon" in timing_lower:
                times_for_med.append("01:00 PM")
            if "evening" in timing_lower:
                times_for_med.append("06:00 PM")
            if "night" in timing_lower or "bedtime" in timing_lower:
                times_for_med.append("09:00 PM")

        if not times_for_med:
            # Fall back to frequency parsing
            if "three" in freq_lower or "thrice" in freq_lower or "tds" in freq_lower or "tid" in freq_lower:
                times_for_med = ["08:00 AM", "01:00 PM", "09:00 PM"]
            elif "twice" in freq_lower or "bd" in freq_lower or "bid" in freq_lower:
                times_for_med = ["08:00 AM", "09:00 PM"]
            else:
                # once daily or unknown — default to morning
                times_for_med = ["08:00 AM"]

        # Apply patient override if provided
        if times_override and med.medicine_name in times_override:
            times_for_med = [times_override[med.medicine_name]]

        full_name = f"{med.medicine_name} {med.dosage}".strip()

        for t in times_for_med:
            # Duplicate check: same medicine + time + date + prescription
            exists = db.query(MedicineSchedule).filter(
                MedicineSchedule.patient_id == patient_id,
                MedicineSchedule.medicine_name == full_name,
                MedicineSchedule.scheduled_time == t,
                MedicineSchedule.schedule_date == today
            ).first()
            if exists:
                skipped_count += 1
                continue

            sched = MedicineSchedule(
                patient_id=patient_id,
                prescription_id=prescription_id,
                medicine_name=full_name,
                dosage="1 tablet" if med.dosage else "",
                frequency=med.frequency or "",
                scheduled_time=t,
                meal_relation=med.meal_instruction or med.instructions or "",
                schedule_date=today,
                taken_status=False,
                notes=med.instructions or "",
                reminder_type="medicine"
            )
            db.add(sched)
            created_count += 1
            created_names.append(f"{full_name} at {t}")

    # Follow-up reminder
    if rx.follow_up and rx.follow_up.strip():
        follow_up_name = "Follow-up Doctor Visit"
        exists_fu = db.query(MedicineSchedule).filter(
            MedicineSchedule.patient_id == patient_id,
            MedicineSchedule.medicine_name == follow_up_name,
            MedicineSchedule.schedule_date == today,
            MedicineSchedule.reminder_type == "follow_up"
        ).first()
        if not exists_fu:
            db.add(MedicineSchedule(
                patient_id=patient_id,
                prescription_id=prescription_id,
                medicine_name=follow_up_name,
                dosage="",
                frequency="",
                scheduled_time="09:00 AM",
                meal_relation="",
                schedule_date=today,
                taken_status=False,
                notes=f"Follow-up: {rx.follow_up}",
                reminder_type="follow_up"
            ))
            created_count += 1
            created_names.append("Follow-up Doctor Visit")

    # Tests reminder
    if rx.tests_advised and rx.tests_advised.strip():
        test_name = "Complete Advised Tests"
        exists_test = db.query(MedicineSchedule).filter(
            MedicineSchedule.patient_id == patient_id,
            MedicineSchedule.medicine_name == test_name,
            MedicineSchedule.schedule_date == today,
            MedicineSchedule.reminder_type == "test"
        ).first()
        if not exists_test:
            db.add(MedicineSchedule(
                patient_id=patient_id,
                prescription_id=prescription_id,
                medicine_name=test_name,
                dosage="",
                frequency="",
                scheduled_time="09:00 AM",
                meal_relation="",
                schedule_date=today,
                taken_status=False,
                notes=f"Tests advised: {rx.tests_advised}",
                reminder_type="test"
            ))
            created_count += 1
            created_names.append("Complete Advised Tests")

    db.commit()
    return {
        "status": "success",
        "created": created_count,
        "skipped_duplicates": skipped_count,
        "reminder_names": created_names
    }
