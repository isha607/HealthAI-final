"""
HealthAI — Phase 7: Context-Aware, Record-Driven AI Health Assistant
===================================================================
A secure, patient-scoped assistant that accesses actual stored records
for the authenticated patient:
- Health Profile
- Laboratory Reports, Parameter History, and Trend Comparisons
- Doctor Prescriptions & Visit History
- Current Medications vs. Past Prescriptions
- Daily Medication Schedules & Reminders
- Tests Advised & Upcoming Follow-Up Consultations
- AI Risk Assessment History (Alzheimer, Diabetes, Cardiovascular)
- Controlled Actions: Reminder creation with validation and database persistence

STRICT CLINICAL SAFETY:
- Emergency triage redirection for acute symptoms
- Strict prohibition of medication alterations or self-prescriptions
- Zero fabrication: Clearly reports when records do not exist
- Clear separation between general healthcare information and personal records
"""

from typing import Dict, Any, List, Optional, Tuple
import re
from datetime import datetime, date
from sqlalchemy.orm import Session

from backend.models.db_models import (
    PatientProfile, 
    LabReport, 
    LabResult, 
    Prescription, 
    Medicine, 
    MedicineSchedule, 
    MLAssessment,
    DoctorVisit
)

EMERGENCY_KEYWORDS = [
    "chest pain", "shortness of breath", "difficulty breathing", "severe dizziness",
    "fainting", "loss of consciousness", "stroke", "paralysis", "heart attack", 
    "coughing blood", "sudden numbness", "slurred speech"
]

MED_ALTERATION_TERMS = [
    "stop taking", "stop my medicine", "should i stop", "quit my medication", 
    "double dose", "change my dose", "increase my dose", "decrease my dose", 
    "prescribe me", "can i take double", "replace my medicine"
]

# Educational lookup for common medications
COMMON_DRUG_KNOWLEDGE = {
    "metformin": "an oral antihyperglycemic medication primarily prescribed for the management of type 2 diabetes. It works by decreasing hepatic glucose production and improving insulin sensitivity.",
    "atorvastatin": "an HMG-CoA reductase inhibitor (statin) used to lower LDL ('bad') cholesterol and triglycerides in the blood, reducing cardiovascular disease risk.",
    "amlodipine": "a calcium channel blocker used to treat hypertension (high blood pressure) and coronary artery disease by relaxing peripheral vascular smooth muscle.",
    "losartan": "an angiotensin II receptor blocker (ARB) used to treat high blood pressure and protect kidney function in diabetic patients.",
    "levothyroxine": "a synthetic thyroid hormone used to treat hypothyroidism (underactive thyroid gland) by restoring adequate T4 hormone levels.",
    "omeprazole": "a proton pump inhibitor (PPI) that decreases the amount of acid produced in the stomach, commonly prescribed for GERD, acid reflux, or gastric ulcer prevention.",
    "aspirin": "an antiplatelet and nonsteroidal anti-inflammatory drug (NSAID) used in low doses for secondary cardiovascular and stroke prevention.",
    "paracetamol": "an analgesic and antipyretic medication widely used for temporary relief of mild-to-moderate pain and fever reduction.",
    "pantoprazole": "a proton pump inhibitor (PPI) that reduces gastric acid secretion, used for acid-related gastrointestinal conditions."
}


# ==============================================================================
# 1. READ-ONLY RECORD RETRIEVAL TOOLS
# ==============================================================================

def get_patient_profile(patient_id: str, db: Session) -> Optional[PatientProfile]:
    return db.query(PatientProfile).filter(PatientProfile.patient_id == patient_id).first()

def get_latest_report(patient_id: str, db: Session) -> Optional[LabReport]:
    return db.query(LabReport).filter(LabReport.patient_id == patient_id).order_by(LabReport.report_date.desc()).first()

def get_report_history(patient_id: str, db: Session) -> List[LabReport]:
    return db.query(LabReport).filter(LabReport.patient_id == patient_id).order_by(LabReport.report_date.desc()).all()

def get_current_prescription(patient_id: str, db: Session) -> Optional[Prescription]:
    return db.query(Prescription).filter(Prescription.patient_id == patient_id).order_by(Prescription.prescription_date.desc()).first()

def get_prescription_history(patient_id: str, db: Session) -> List[Prescription]:
    return db.query(Prescription).filter(Prescription.patient_id == patient_id).order_by(Prescription.prescription_date.desc()).all()

def get_today_reminders(patient_id: str, db: Session) -> List[MedicineSchedule]:
    today_str = date.today().isoformat()
    # Returns today's schedules or recurring/empty date schedules
    return db.query(MedicineSchedule).filter(
        MedicineSchedule.patient_id == patient_id,
        (MedicineSchedule.schedule_date == today_str) | (MedicineSchedule.schedule_date == "") | (MedicineSchedule.schedule_date == None)
    ).all()

def get_all_reminders(patient_id: str, db: Session) -> List[MedicineSchedule]:
    return db.query(MedicineSchedule).filter(MedicineSchedule.patient_id == patient_id).all()

def get_assessment_history(patient_id: str, db: Session) -> List[MLAssessment]:
    return db.query(MLAssessment).filter(MLAssessment.patient_id == patient_id).order_by(MLAssessment.created_at.desc()).all()

def get_upcoming_doctor_visit(patient_id: str, db: Session) -> Optional[DoctorVisit]:
    return db.query(DoctorVisit).filter(DoctorVisit.patient_id == patient_id).order_by(DoctorVisit.visit_date.desc()).first()


def search_parameter_across_reports(patient_id: str, term: str, db: Session) -> List[Tuple[date, LabResult, str]]:
    """
    Finds all historical lab results matching a parameter name (e.g., TSH, glucose, cholesterol, hemoglobin),
    ordered from newest to oldest.
    """
    reports = db.query(LabReport).filter(LabReport.patient_id == patient_id).order_by(LabReport.report_date.desc()).all()
    matches = []
    term_lower = term.lower()

    for rep in reports:
        for r in rep.results:
            tname = r.test_name.lower()
            if term_lower in tname:
                matches.append((rep.report_date, r, rep.report_title))
                break

    return matches


# ==============================================================================
# 2. CONTROLLED ACTION: CREATE REMINDER
# ==============================================================================

def extract_and_create_reminder(patient_id: str, query: str, db: Session) -> Optional[Dict[str, Any]]:
    """
    Controlled tool action: Identifies medicine name, scheduled time, and meal relation
    from an explicit patient request (e.g. "Remind me to take Metformin at 8 PM after dinner").
    Safely creates a new MedicineSchedule entry in the database.
    """
    q_lower = query.lower()
    
    # Must contain an explicit reminder intent
    if not any(k in q_lower for k in ["remind me", "set a reminder", "create reminder", "schedule a reminder", "add reminder"]):
        return None

    # 1. Extract Time (e.g. 8 PM, 08:00 AM, 9:30 pm, 20:00)
    time_match = re.search(r"\b([0-1]?[0-9]|2[0-3])(?::([0-5][0-9]))?\s*(am|pm)?\b", q_lower)
    time_str = "08:00 AM" # default if unspecified
    if time_match:
        hr = int(time_match.group(1))
        minute = time_match.group(2) or "00"
        meridiem = time_match.group(3)
        if meridiem:
            time_str = f"{hr:02d}:{minute} {meridiem.upper()}"
        else:
            # 24-hr to 12-hr conversion
            if hr >= 12:
                time_str = f"{(hr-12 if hr>12 else 12):02d}:{minute} PM"
            else:
                time_str = f"{(hr if hr>0 else 12):02d}:{minute} AM"

    # 2. Extract Meal Relation
    meal_rel = "After food"
    if "before" in q_lower:
        meal_rel = "Before food"
    elif "with" in q_lower:
        meal_rel = "With food"
    elif "empty stomach" in q_lower:
        meal_rel = "Empty stomach"

    # 3. Extract Medicine Name
    # First check against the patient's existing known prescribed medicines
    known_meds = db.query(Medicine.medicine_name).filter(Medicine.patient_id == patient_id).distinct().all()
    med_name = ""
    for (m,) in known_meds:
        if m.lower() in q_lower:
            med_name = m
            break

    # If not matching known meds, extract candidate words after 'take' or 'for'
    if not med_name:
        candidate_match = re.search(r"(?:take|for|remind me to take)\s+([a-zA-Z0-9\-\s]{3,30}?)(?:\s+at|\s+around|\s+after|\s+before|\s+every|$)", query, re.IGNORECASE)
        if candidate_match:
            cand = candidate_match.group(1).strip()
            # Filter out non-medicine stop words
            if cand.lower() not in ["my medicine", "medicine", "pill", "pills", "tablet", "tablets", "dose"]:
                med_name = cand.title()

    if not med_name:
        med_name = "Prescribed Medication"

    today_str = date.today().isoformat()

    # Create new controlled reminder entry
    new_rem = MedicineSchedule(
        patient_id=patient_id,
        medicine_name=med_name,
        dosage="1 dose",
        scheduled_time=time_str,
        meal_relation=meal_rel,
        schedule_date=today_str,
        taken_status=False,
        notes="Scheduled via HealthAI Assistant",
        frequency="Once daily",
        reminder_type="medicine"
    )
    db.add(new_rem)
    db.commit()
    db.refresh(new_rem)

    return {
        "success": True,
        "reminder_id": new_rem.id,
        "medicine_name": med_name,
        "scheduled_time": time_str,
        "meal_relation": meal_rel,
        "date": today_str
    }


# ==============================================================================
# 3. MAIN QUERY PROCESSOR
# ==============================================================================

def handle_chatbot_query(patient_id: str, query: str, db: Session) -> Dict[str, Any]:
    """
    Context-aware HealthAI Assistant scoped strictly to the authenticated patient's records.
    """
    q_clean = query.strip()
    q_lower = q_clean.lower()
    referenced_context = []

    # --------------------------------------------------------------------------
    # Rule 1: Emergency Triage Redirection
    # --------------------------------------------------------------------------
    for kw in EMERGENCY_KEYWORDS:
        if kw in q_lower:
            return {
                "response": (
                    "⚠️ **EMERGENCY MEDICAL ADVISORY**:\n\n"
                    "Your message mentions symptoms (such as chest pain, severe shortness of breath, sudden numbness, or acute distress) "
                    "that require immediate medical evaluation. **Do not wait for an automated reply.**\n\n"
                    "• Immediately call your local emergency service (e.g. **911**, **112**, or **108**).\n"
                    "• Have someone take you to the nearest hospital emergency room immediately."
                ),
                "referenced_context": ["Emergency Safety Protocol"],
                "disclaimer": "Emergency triage automated advisory."
            }

    # --------------------------------------------------------------------------
    # Rule 2: Strict Prohibition of Medication Alterations
    # --------------------------------------------------------------------------
    for term in MED_ALTERATION_TERMS:
        if term in q_lower:
            return {
                "response": (
                    "🛡️ **Medication Safety Rule**:\n\n"
                    "I cannot advise you to start, alter, adjust, or stop any prescribed medication or dosage. "
                    "Modifying your medication regimen without medical supervision can lead to serious adverse effects or drug interactions.\n\n"
                    "Please contact your prescribing physician or pharmacist before making any changes to your treatment."
                ),
                "referenced_context": ["Medication Safety Policy"],
                "disclaimer": "HealthAI Assistant is a decision-support and records-tracking tool. It cannot prescribe or modify medical treatment."
            }

    # --------------------------------------------------------------------------
    # Rule 3: Controlled Action: Setting a Reminder
    # --------------------------------------------------------------------------
    if any(k in q_lower for k in ["remind me", "set a reminder", "create reminder", "schedule reminder", "add reminder"]):
        action_res = extract_and_create_reminder(patient_id, query, db)
        if action_res and action_res.get("success"):
            return {
                "response": (
                    f"✅ **Reminder Successfully Created**\n\n"
                    f"• **Medication**: {action_res['medicine_name']}\n"
                    f"• **Scheduled Time**: {action_res['scheduled_time']}\n"
                    f"• **Meal Relation**: {action_res['meal_relation']}\n"
                    f"• **Date**: Today ({action_res['date']})\n\n"
                    "This reminder has been added to your daily schedule and is now visible in your **Reminders** page and Dashboard."
                ),
                "referenced_context": ["Medicine Schedules (Database Write Action)"],
                "disclaimer": "Automated schedule notification tool. Take medications as directed by your physician."
            }

    # --------------------------------------------------------------------------
    # Rule 3.5: General Educational Drug Information
    # --------------------------------------------------------------------------
    for drug_name, drug_info in COMMON_DRUG_KNOWLEDGE.items():
        if re.search(r"\b" + re.escape(drug_name) + r"\b", q_lower) and any(k in q_lower for k in ["used for", "what is", "why is", "tell me about", "info", "information", "purpose", "general"]):
            return {
                "response": (
                    f"📖 **General Information for {drug_name.capitalize()}**:\n\n"
                    f"{drug_name.capitalize()} is {drug_info}\n\n"
                    "*(Note: This is general educational pharmaceutical information. Always follow the specific dosage, timing, and directions from your prescribing doctor.)*"
                ),
                "referenced_context": ["General Clinical Pharmacology Knowledge"],
                "disclaimer": "Educational information. Does not replace physician consultation."
            }

    # --------------------------------------------------------------------------
    # Rule 3.8: Whole-Report Longitudinal Comparison
    # --------------------------------------------------------------------------
    if any(k in q_lower for k in ["what changed", "changes between", "compare my reports", "compare my previous and latest", "compare reports", "compare my latest report"]):
        referenced_context.append("Lab Report Longitudinal History")
        reports = get_report_history(patient_id, db)
        reports_with_results = [rep for rep in reports if rep.results and len(rep.results) > 0]
        if len(reports_with_results) >= 2:
            latest = reports_with_results[0]
            prev = reports_with_results[1]
            prev_map = {r.test_name.lower(): r for r in prev.results}
            comparison_lines = []
            
            for r_lat in latest.results:
                k = r_lat.test_name.lower()
                if k in prev_map:
                    r_prv = prev_map[k]
                    if r_prv.result_value > 0 and r_lat.result_value > 0:
                        diff = round(r_lat.result_value - r_prv.result_value, 2)
                        sign = "+" if diff > 0 else ""
                        if diff != 0:
                            change_type = "increased" if diff > 0 else "decreased"
                            comparison_lines.append(f"• **{r_lat.test_name}**: {change_type} from {r_prv.result_value} to **{r_lat.result_value} {r_lat.unit}** ({sign}{diff} {r_lat.unit})")
                        else:
                            comparison_lines.append(f"• **{r_lat.test_name}**: unchanged at **{r_lat.result_value} {r_lat.unit}**")
                    elif r_lat.result_value > 0 and r_prv.result_value <= 0:
                        comparison_lines.append(f"• **{r_lat.test_name}**: recorded as **{r_lat.result_value} {r_lat.unit}** (unrecorded in previous report)")

            if comparison_lines:
                return {
                    "response": (
                        f"📊 **Comparison between your two latest reports**:\n\n"
                        f"• **Previous**: *{prev.report_title}* ({prev.report_date.strftime('%B %d, %Y')})\n"
                        f"• **Latest**: *{latest.report_title}* ({latest.report_date.strftime('%B %d, %Y')})\n\n"
                        "**Parameter Changes**:\n" +
                        "\n".join(comparison_lines)
                    ),
                    "referenced_context": referenced_context,
                    "disclaimer": "Longitudinal report comparison."
                }
            else:
                return {
                    "response": f"Your latest report (*{latest.report_title}*, {latest.report_date.strftime('%b %d, %Y')}) and previous report (*{prev.report_title}*, {prev.report_date.strftime('%b %d, %Y')}) measured different diagnostic parameters, so direct numerical comparisons cannot be calculated.",
                    "referenced_context": referenced_context,
                    "disclaimer": "Diagnostic parameter matching."
                }
        else:
            return {
                "response": "At least two completed lab reports with test results are required to compare report changes. You have only one completed report on record.",
                "referenced_context": referenced_context,
                "disclaimer": "Zero fabrication."
            }

    # --------------------------------------------------------------------------
    # Rule 4: Parameter Comparison Across Reports (e.g. "Compare my TSH", "TSH higher or lower")
    # --------------------------------------------------------------------------
    param_keywords = ["tsh", "glucose", "cholesterol", "hdl", "ldl", "triglyceride", "hba1c", "hemoglobin", "platelet", "wbc", "creatinine", "blood pressure", "bp", "vitamin", "calcium", "potassium", "sodium", "uric acid", "bilirubin", "sgot", "sgpt", "alt", "ast"]
    target_param = None
    for pk in param_keywords:
        if re.search(r"\b" + re.escape(pk) + r"\b", q_lower):
            target_param = pk
            break

    # If not in fixed list, check for questions like "what is my [xyz] level/value/result?"
    if not target_param:
        param_match = re.search(r"(?:what(?:'s|\s+is)\s+my\s+(?:latest\s+)?|check\s+my\s+|my\s+latest\s+)([a-zA-Z0-9\s\-]+?)(?:\s+level|\s+value|\s+test|\s+result|\s+count|\?|$)", q_lower)
        if param_match:
            cand = param_match.group(1).strip()
            # Must not be an inquiry about risk, medicines, doctors, reminders, etc.
            if cand and not any(stop in cand for stop in ["medicine", "prescription", "doctor", "profile", "history", "reminder", "schedule", "risk", "health", "assessment", "score", "visit"]):
                target_param = cand

    is_comparison_query = any(k in q_lower for k in ["compare", "higher", "lower", "change", "increased", "decreased", "previous report", "earlier", "before"])

    if target_param and is_comparison_query:
        referenced_context.append("Historical Lab Reports (Parameter Comparison)")
        history = search_parameter_across_reports(patient_id, target_param, db)
        
        if len(history) >= 2:
            latest_date, latest_res, latest_title = history[0]
            # Pick the most recent previous result from a different report date or report title
            prev_entry = None
            for entry in history[1:]:
                if entry[0] != latest_date or entry[2] != latest_title:
                    prev_entry = entry
                    break
            if not prev_entry:
                prev_entry = history[1]

            prev_date, prev_res, prev_title = prev_entry
            diff = round(latest_res.result_value - prev_res.result_value, 2)
            pct = round((diff / prev_res.result_value) * 100, 1) if prev_res.result_value else 0
            
            trend_str = "an increase" if diff > 0 else ("a decrease" if diff < 0 else "no change")
            trend_sign = "+" if diff > 0 else ""

            return {
                "response": (
                    f"📊 **Report Comparison for {latest_res.test_name}**:\n\n"
                    f"• **Previous Test** ({prev_date.strftime('%b %d, %Y')} in *{prev_title}*):\n"
                    f"  **{prev_res.result_value} {prev_res.unit}** (Status: {prev_res.status})\n\n"
                    f"• **Latest Test** ({latest_date.strftime('%b %d, %Y')} in *{latest_title}*):\n"
                    f"  **{latest_res.result_value} {latest_res.unit}** (Status: {latest_res.status})\n\n"
                    f"**Analysis**: That is {trend_str} of **{trend_sign}{diff} {latest_res.unit}** ({trend_sign}{pct}%) between the two reports.\n"
                    f"Standard laboratory reference range: {latest_res.reference_low or '—'} to {latest_res.reference_high or '—'} {latest_res.unit}."
                ),
                "referenced_context": referenced_context,
                "disclaimer": "Diagnostic parameter comparison based on verified uploaded patient laboratory documents."
            }
        elif len(history) == 1:
            r_date, r_val, r_title = history[0]
            return {
                "response": (
                    f"According to your records, only one test result is available for **{r_val.test_name}**:\n\n"
                    f"• **Date**: {r_date.strftime('%B %d, %Y')} (*{r_title}*)\n"
                    f"• **Value**: **{r_val.result_value} {r_val.unit}** ({r_val.status})\n"
                    f"• **Reference Range**: {r_val.reference_low or '—'} – {r_val.reference_high or '—'} {r_val.unit}\n\n"
                    "An earlier test is not on record to calculate a historical comparison. Upload an earlier report in the Lab Reports tab to track trends over time."
                ),
                "referenced_context": referenced_context,
                "disclaimer": "Educational record lookup."
            }
        else:
            return {
                "response": f"I checked your stored laboratory records, and no test results for **'{target_param.upper()}'** were found. You have not uploaded a report containing this parameter yet.",
                "referenced_context": ["Laboratory Records"],
                "disclaimer": "Zero fabrication guarantee."
            }

    # --------------------------------------------------------------------------
    # Rule 5: Specific Parameter Query (e.g. "What's my latest TSH?")
    # --------------------------------------------------------------------------
    if target_param:
        referenced_context.append("Latest Lab Reports")
        history = search_parameter_across_reports(patient_id, target_param, db)
        if history:
            latest_date, r_val, r_title = history[0]
            status_note = f" (Status: **{r_val.status}**)" if r_val.status != "NORMAL" else " (Status: Normal)"
            return {
                "response": (
                    f"📋 **Latest {r_val.test_name} Result**:\n\n"
                    f"• **Value**: **{r_val.result_value} {r_val.unit}**{status_note}\n"
                    f"• **Reference Range**: {r_val.reference_low or 0} to {r_val.reference_high or '—'} {r_val.unit}\n"
                    f"• **Report**: *{r_title}*\n"
                    f"• **Report Date**: {latest_date.strftime('%B %d, %Y')}\n\n"
                    f"Clinical interpretation: {r_val.clinical_explanation or 'Value is within expected reference bounds.'}"
                ),
                "referenced_context": referenced_context,
                "disclaimer": "Laboratory report values extracted from verified patient uploads."
            }
        else:
            return {
                "response": f"I checked your stored laboratory records, and no test results for **'{target_param.upper()}'** were found in your uploaded reports.",
                "referenced_context": ["Laboratory Records"],
                "disclaimer": "Zero fabrication policy."
            }

    # --------------------------------------------------------------------------
    # Rule 6: General Lab Reports Overview & Abnormal Results
    # --------------------------------------------------------------------------
    if any(k in q_lower for k in ["lab report", "blood report", "test report", "abnormal", "blood test", "cbc"]):
        referenced_context.append("Lab Reports")
        latest_report = get_latest_report(patient_id, db)
        if latest_report:
            abnormal = [r for r in latest_report.results if r.status in ["HIGH", "LOW"]]
            if abnormal:
                lines = [f"• **{r.test_name}**: {r.result_value} {r.unit} (**{r.status}**) — Ref: {r.reference_low or 0}–{r.reference_high or '—'} {r.unit}" for r in abnormal]
                return {
                    "response": (
                        f"In your most recent lab report (**{latest_report.report_title}** from {latest_report.report_date.strftime('%B %d, %Y')}), "
                        f"the following parameters were outside standard reference limits:\n\n" +
                        "\n".join(lines) +
                        "\n\n*Please discuss any abnormal values with your healthcare provider for clinical evaluation.*"
                    ),
                    "referenced_context": referenced_context,
                    "disclaimer": "Record extraction from patient lab documents."
                }
            else:
                return {
                    "response": f"In your latest report (**{latest_report.report_title}** from {latest_report.report_date.strftime('%B %d, %Y')}), all {len(latest_report.results)} measured parameters are within standard normal reference ranges.",
                    "referenced_context": referenced_context,
                    "disclaimer": "Record extraction."
                }
        else:
            return {
                "response": "You haven't uploaded any lab reports yet. You can upload diagnostic reports in the **Lab Reports** tab to automatically extract and monitor your parameters.",
                "referenced_context": referenced_context,
                "disclaimer": "Zero fabrication."
            }

    # --------------------------------------------------------------------------
    # Rule 7: Prescriptions & Doctor Visits
    # --------------------------------------------------------------------------
    if any(k in q_lower for k in [
        "prescription", "prescriptions", "doctor prescribe", "prescribing doctor", 
        "my doctor", "who is my doctor", "which doctor", "prescribed by", 
        "last visit", "doctor visit", "clinical note"
    ]):
        referenced_context.append("Prescriptions")
        latest_rx = get_current_prescription(patient_id, db)
        if latest_rx:
            med_lines = []
            for m in latest_rx.medicines:
                timing_str = f", {m.timing}" if m.timing else ""
                meal_str = f" ({m.meal_instruction})" if m.meal_instruction else ""
                med_lines.append(f"• **{m.medicine_name}** — {m.dosage}, {m.frequency}{timing_str}{meal_str}")
            
            notes_str = f"\n• **Doctor Notes / Diagnosis**: {latest_rx.diagnosis}" if latest_rx.diagnosis else ""
            tests_str = f"\n• **Tests Advised**: {latest_rx.tests_advised}" if latest_rx.tests_advised else ""
            follow_str = f"\n• **Follow-Up**: {latest_rx.follow_up}" if latest_rx.follow_up else ""

            return {
                "response": (
                    f"📝 **Latest Doctor Prescription Details**:\n\n"
                    f"• **Doctor**: {latest_rx.doctor_name or 'Prescribing Physician'}\n"
                    f"• **Date**: {latest_rx.prescription_date.strftime('%B %d, %Y')}"
                    f"{notes_str}\n\n"
                    f"**Prescribed Medicines** ({len(latest_rx.medicines)}):\n" +
                    "\n".join(med_lines) +
                    f"{tests_str}{follow_str}"
                ),
                "referenced_context": referenced_context,
                "disclaimer": "Prescription details extracted from verified clinical records."
            }
        else:
            return {
                "response": "No prescriptions are currently on record for your profile. You can upload prescription documents or photos in the **Prescriptions** tab.",
                "referenced_context": referenced_context,
                "disclaimer": "Zero fabrication."
            }

    # --------------------------------------------------------------------------
    # Rule 8: Current Medicines vs Past Medicines
    # --------------------------------------------------------------------------
    if any(k in q_lower for k in [
        "medicines taking", "current medicine", "my medicine", "active medication", 
        "which medicine", "what medicines am i", "medication", "medications", 
        "my medications", "current medications", "what medications am i", "pills am i", 
        "medicines am i", "prescribed medicine", "prescribed medication"
    ]):
        referenced_context.append("Prescriptions & Medicines")
        latest_rx = get_current_prescription(patient_id, db)
        if latest_rx and latest_rx.medicines:
            lines = []
            for m in latest_rx.medicines:
                extra = []
                if m.frequency: extra.append(m.frequency)
                if m.timing: extra.append(m.timing)
                if m.meal_instruction: extra.append(m.meal_instruction)
                detail = f" — {', '.join(extra)}" if extra else ""
                lines.append(f"• **{m.medicine_name}** ({m.dosage}){detail}")

            return {
                "response": (
                    f"💊 **Current Prescribed Medications** (from your latest prescription on {latest_rx.prescription_date.strftime('%B %d, %Y')}):\n\n" +
                    "\n".join(lines) +
                    "\n\n*Always follow the exact timing and instructions provided by your prescribing doctor.*"
                ),
                "referenced_context": referenced_context,
                "disclaimer": "Records retrieved from your latest confirmed doctor prescription."
            }
        else:
            return {
                "response": "You do not have any active prescription medicines currently recorded. You can add or upload prescriptions in the Prescriptions tab.",
                "referenced_context": referenced_context,
                "disclaimer": "Zero fabrication."
            }

    # --------------------------------------------------------------------------
    # Rule 9: Reminders / Today's Doses
    # --------------------------------------------------------------------------
    if any(k in q_lower for k in ["reminder", "doses today", "take today", "pending", "taken", "schedule"]):
        referenced_context.append("Medicine Reminders")
        rems = get_today_reminders(patient_id, db)
        if rems:
            taken = [r for r in rems if r.taken_status]
            pending = [r for r in rems if not r.taken_status]

            lines = []
            for r in rems:
                status_icon = "✅ Taken" if r.taken_status else "⏳ Pending"
                meal = f" • {r.meal_relation}" if r.meal_relation else ""
                lines.append(f"• **{r.scheduled_time}**: {r.medicine_name} ({r.dosage}{meal}) — *{status_icon}*")

            return {
                "response": (
                    f"⏰ **Today's Medication Tracker** ({len(taken)} taken, {len(pending)} pending):\n\n" +
                    "\n".join(lines) +
                    "\n\nYou can mark doses as completed directly in the **Reminders** tab."
                ),
                "referenced_context": referenced_context,
                "disclaimer": "Daily compliance tracker."
            }
        else:
            return {
                "response": "You have no active reminders scheduled for today. You can ask me to set one (e.g. *'Remind me to take Metformin at 8 PM'*) or configure them in the Reminders tab.",
                "referenced_context": referenced_context,
                "disclaimer": "Zero fabrication."
            }

    # --------------------------------------------------------------------------
    # Rule 10: Follow-Up & Tests Advised
    # --------------------------------------------------------------------------
    if any(k in q_lower for k in ["follow up", "follow-up", "next visit", "tests advised", "tests to do", "tests do i need"]):
        referenced_context.append("Prescriptions & Doctor Consultations")
        rxs = get_prescription_history(patient_id, db)
        
        found_followup = None
        found_tests = None
        doc_name = "your doctor"
        rx_date_str = ""

        for rx in rxs:
            if not found_followup and rx.follow_up and rx.follow_up.strip():
                found_followup = rx.follow_up.strip()
                doc_name = rx.doctor_name or doc_name
                rx_date_str = rx.prescription_date.strftime('%B %d, %Y')
            if not found_tests and rx.tests_advised and rx.tests_advised.strip():
                found_tests = rx.tests_advised.strip()
                if not rx_date_str:
                    rx_date_str = rx.prescription_date.strftime('%B %d, %Y')
            if found_followup and found_tests:
                break

        # Also check DoctorVisit if follow-up wasn't in prescription
        if not found_followup:
            visit = get_upcoming_doctor_visit(patient_id, db)
            if visit and hasattr(visit, "follow_up_date") and visit.follow_up_date:
                found_followup = visit.follow_up_date.strftime('%B %d, %Y')
                doc_name = visit.doctor_name or doc_name

        if found_followup or found_tests:
            parts = []
            if found_followup:
                parts.append(f"• **Next Follow-up Consultation**: {found_followup} (with {doc_name})")
            if found_tests:
                parts.append(f"• **Advised Tests**: {found_tests}")
            
            date_info = f" (from prescription dated {rx_date_str})" if rx_date_str else ""
            return {
                "response": (
                    f"🗓️ **Clinical Follow-up & Advised Tests**{date_info}:\n\n" +
                    "\n".join(parts)
                ),
                "referenced_context": referenced_context,
                "disclaimer": "Directives extracted from doctor prescriptions."
            }
        else:
            return {
                "response": "No pending follow-up date or advised diagnostic tests are currently recorded in your prescriptions.",
                "referenced_context": referenced_context,
                "disclaimer": "Zero fabrication."
            }

    # --------------------------------------------------------------------------
    # Rule 11: AI Risk Assessment History
    # --------------------------------------------------------------------------
    if any(k in q_lower for k in ["risk", "alzheimer", "diabetes assessment", "cardio assessment", "framingham", "ai assessment", "risk score"]):
        referenced_context.append("AI Risk Assessment History")
        assessments = get_assessment_history(patient_id, db)
        if assessments:
            # Deduplicate by model type
            seen_types = set()
            lines = []
            for a in assessments:
                raw_type = a.assessment_type.lower()
                if "framingham" in raw_type or "cardio" in raw_type:
                    atype = "Cardiovascular (Framingham)"
                elif "alz" in raw_type:
                    atype = "Alzheimer's Disease"
                elif "diab" in raw_type:
                    atype = "Diabetes"
                else:
                    atype = a.assessment_type.capitalize()

                if atype not in seen_types:
                    seen_types.add(atype)
                    src = f" (Source: {a.source_type.replace('_', ' ').title()})" if hasattr(a, "source_type") and a.source_type else ""
                    lines.append(f"• **{atype}**: **{a.risk_level}** ({a.probability}% statistical risk score){src}")

            return {
                "response": (
                    "🧠 **Recent AI Risk Assessment Scores**:\n\n" +
                    "\n".join(lines) +
                    "\n\n*Clinical Notice: All risk predictions are statistical decision-support calculations based on verified health markers. They are not medical diagnoses.*"
                ),
                "referenced_context": referenced_context,
                "disclaimer": "Statistical machine learning model results."
            }
        else:
            return {
                "response": "No AI risk assessments have been run for your profile yet. You can evaluate your risk in the **AI Risk Assessment** tab using manual input or by uploading a medical report.",
                "referenced_context": referenced_context,
                "disclaimer": "Zero fabrication."
            }

    # --------------------------------------------------------------------------
    # Rule 12: Health Profile Information
    # --------------------------------------------------------------------------
    if any(k in q_lower for k in ["profile", "blood group", "allergies", "conditions", "my age", "my weight", "my height", "my bmi"]):
        referenced_context.append("Health Profile")
        profile = get_patient_profile(patient_id, db)
        if profile:
            lines = [
                f"• **Patient Name**: {profile.full_name}",
                f"• **Age / Gender**: {profile.age or 'Not recorded'} years / {profile.gender or 'Not recorded'}",
                f"• **Blood Group**: {profile.blood_group or 'Not recorded'}",
                f"• **Height & Weight**: {profile.height_cm or '—'} cm, {profile.weight_kg or '—'} kg",
                f"• **Allergies**: {profile.allergies or 'None recorded'}",
                f"• **Medical Conditions**: {profile.existing_conditions or 'None recorded'}"
            ]
            return {
                "response": "👤 **Your Health Profile Summary**:\n\n" + "\n".join(lines),
                "referenced_context": referenced_context,
                "disclaimer": "Patient record data."
            }

    # --------------------------------------------------------------------------
    # Rule 13: Medical History Summary (Synthesized)
    # --------------------------------------------------------------------------
    if any(k in q_lower for k in ["summarize my", "health summary", "medical history", "recent history", "overview of my health"]):
        referenced_context.append("Comprehensive Health Record")
        profile = get_patient_profile(patient_id, db)
        latest_report = get_latest_report(patient_id, db)
        latest_rx = get_current_prescription(patient_id, db)
        assessments = get_assessment_history(patient_id, db)
        rems = get_today_reminders(patient_id, db)

        summary_parts = []
        if profile:
            summary_parts.append(f"**Patient**: {profile.full_name} (Age: {profile.age or '—'}, Blood Group: {profile.blood_group or '—'})")
        
        if latest_report:
            abnormal = [r for r in latest_report.results if r.status in ["HIGH", "LOW"]]
            ab_str = f"{len(abnormal)} abnormal value(s) requiring attention" if abnormal else "all measured parameters normal"
            summary_parts.append(f"**Latest Lab Report** ({latest_report.report_date.strftime('%b %d, %Y')}): *{latest_report.report_title}* — {ab_str}")

        if latest_rx:
            summary_parts.append(f"**Current Treatment** (Dr. {latest_rx.doctor_name or 'Physician'}, {latest_rx.prescription_date.strftime('%b %d, %Y')}): {len(latest_rx.medicines)} active medication(s)")
            if latest_rx.follow_up:
                summary_parts.append(f"**Upcoming Follow-up**: {latest_rx.follow_up}")

        if assessments:
            top_a = assessments[0]
            summary_parts.append(f"**Latest AI Risk Scan**: {top_a.assessment_type.capitalize()} ({top_a.risk_level}, {top_a.probability}%)")

        if rems:
            summary_parts.append(f"**Daily Regimen**: {len(rems)} scheduled reminder(s) configured for today")

        return {
            "response": (
                "🏥 **Comprehensive Health History Summary**:\n\n" +
                "\n\n".join(summary_parts) +
                "\n\n*Ask me about any specific report, parameter, medication, or reminder for in-depth details.*"
            ),
            "referenced_context": referenced_context,
            "disclaimer": "Synthesized clinical record summary."
        }

    # --------------------------------------------------------------------------
    # Rule 14: General Healthcare Knowledge
    # --------------------------------------------------------------------------
    for drug_name, drug_info in COMMON_DRUG_KNOWLEDGE.items():
        if drug_name in q_lower:
            return {
                "response": (
                    f"📖 **General Information for {drug_name.capitalize()}**:\n\n"
                    f"{drug_name.capitalize()} is {drug_info}\n\n"
                    "*(Note: This is general educational pharmaceutical information. Always follow the specific dosage, timing, and directions from your prescribing doctor.)*"
                ),
                "referenced_context": ["General Clinical Pharmacology Knowledge"],
                "disclaimer": "Educational information. Does not replace physician consultation."
            }

    # --------------------------------------------------------------------------
    # Default Welcome & Guided Prompt
    # --------------------------------------------------------------------------
    profile = get_patient_profile(patient_id, db)
    name = profile.full_name if profile else "there"
    return {
        "response": (
            f"Hello {name}! I am your personal HealthAI Assistant, connected directly to your health records.\n\n"
            "Here are some questions you can ask me:\n"
            "• *'What is my latest TSH or fasting glucose?'*\n"
            "• *'Compare my TSH with my previous report.'*\n"
            "• *'What medicines am I currently taking?'*\n"
            "• *'When is my next doctor follow-up?'*\n"
            "• *'What were the tests advised by my doctor?'*\n"
            "• *'Remind me to take my medicine at 8 PM.'*\n"
            "• *'Summarize my recent health history.'*"
        ),
        "referenced_context": ["Patient Profile"],
        "disclaimer": "Educational health records assistant. Not a medical doctor."
    }
