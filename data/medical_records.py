"""
Comprehensive mock data for the Healthcare Claims HITL Review Pipeline.

Simulates:
  - Patient profiles (patient DB)
  - Medical documents / PDFs (document store)
  - Historical claims (claims DB)
  - Treatment guidelines (guidelines DB)

Each PDF record has `content` that mimics realistic extracted text from a real
medical document.  The content is deliberately verbose so that the "extract
relevant sections before LLM call" optimisation produces a meaningful token saving.
"""

from __future__ import annotations
from typing import Dict, List, Any

# ══════════════════════════════════════════════════════════════════════════════
# PATIENT PROFILES
# ══════════════════════════════════════════════════════════════════════════════

PATIENT_PROFILES: Dict[str, Dict[str, Any]] = {

    "P001": {
        "id": "P001",
        "name": "John Doe",
        "dob": "1972-03-15",
        "age": 52,
        "gender": "Male",
        "conditions": ["Osteoarthritis (right knee, Grade III)", "Hypertension"],
        "medications": ["Meloxicam 15 mg daily", "Lisinopril 10 mg daily"],
        "allergies": ["Penicillin"],
        "primary_care": "Dr. Sarah Mitchell",
        "insurance_id": "INS-P001",
        "plan": "PPO Gold",
        "deductible_met": True,
        "out_of_pocket_remaining": 1200.00,
    },

    "P002": {
        "id": "P002",
        "name": "Mary Johnson",
        "dob": "1989-07-22",
        "age": 35,
        "gender": "Female",
        "conditions": [],
        "medications": [],
        "allergies": [],
        "primary_care": "Dr. Alan Torres",
        "insurance_id": "INS-P002",
        "plan": "HMO Silver",
        "deductible_met": False,
        "out_of_pocket_remaining": 3500.00,
    },

    "P003": {
        "id": "P003",
        "name": "Michael Chen",
        "dob": "1963-11-08",
        "age": 61,
        "gender": "Male",
        "conditions": ["Non-Hodgkin Lymphoma (Stage III, in treatment)", "Type 2 Diabetes"],
        "medications": ["Metformin 1000 mg twice daily", "Ondansetron 8 mg PRN"],
        "allergies": ["Sulfa drugs"],
        "primary_care": "Dr. Rebecca Nguyen",
        "insurance_id": "INS-P003",
        "plan": "PPO Platinum",
        "deductible_met": True,
        "out_of_pocket_remaining": 0.00,   # max OOP reached
    },

    "P004": {
        "id": "P004",
        "name": "Sarah Williams",
        "dob": "1986-02-14",
        "age": 38,
        "gender": "Female",
        "conditions": ["Lumbar Degenerative Disc Disease (L4-L5, L5-S1)", "Chronic low-back pain"],
        "medications": ["Gabapentin 300 mg three times daily", "Cyclobenzaprine 5 mg PRN"],
        "allergies": ["Codeine"],
        "primary_care": "Dr. James Peterson",
        "insurance_id": "INS-P004",
        "plan": "PPO Gold",
        "deductible_met": True,
        "out_of_pocket_remaining": 800.00,
    },

    "P005": {
        "id": "P005",
        "name": "Robert Davis",
        "dob": "1997-05-30",
        "age": 27,
        "gender": "Male",
        "conditions": [],
        "medications": [],
        "allergies": [],
        "primary_care": "Dr. Lisa Park",
        "insurance_id": "INS-P005",
        "plan": "HMO Bronze",
        "deductible_met": False,
        "out_of_pocket_remaining": 6000.00,
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# MEDICAL PDF RECORDS
# Each entry has:
#   doc_id, filename, type, pages, content (extracted text - intentionally long)
# ══════════════════════════════════════════════════════════════════════════════

MEDICAL_PDF_RECORDS: Dict[str, List[Dict[str, Any]]] = {

    # ── P001 – John Doe ─────────────────────────────────────────────────────
    "P001": [
        {
            "doc_id": "DOC-P001-001",
            "filename": "mri_right_knee_2026_06_15.pdf",
            "type": "radiology_report",
            "pages": 4,
            "content": """
METROPOLITAN IMAGING CENTER
Patient: John Doe  |  DOB: 1972-03-15  |  MRN: MRN-001234
Referring Physician: Dr. Sarah Mitchell, MD
Exam Date: June 15, 2026  |  Report Date: June 16, 2026
Exam: MRI Right Knee Without Contrast  |  Accession: ACC-78234

CLINICAL HISTORY:
52-year-old male with 18-month history of progressive right knee pain limiting ambulation.
Prior conservative treatment: physical therapy (6 months, ended February 2026),
NSAID therapy (Meloxicam 15 mg), two corticosteroid injections (October 2025, February 2026).
Presenting for surgical planning evaluation.

TECHNIQUE:
MRI of the right knee performed at 3.0T field strength without intravenous contrast.
Sagittal PD-fat-sat, coronal PD-fat-sat, axial PD-fat-sat, sagittal T1, coronal T2,
axial T2 sequences obtained.

FINDINGS:
Medial compartment:
  - Grade III chondromalacia of the medial femoral condyle with full-thickness cartilage
    loss over a 1.5 cm × 1.2 cm region at the weight-bearing surface.
  - Medial meniscus: Horizontal tear in the posterior horn with meniscal extrusion 3.1 mm.
  - Joint space narrowing: 2.0 mm (normal 4–5 mm).
  - Subchondral sclerosis and small marginal osteophytes at medial compartment.

Lateral compartment:
  - Grade II chondromalacia of the lateral femoral condyle, partial thickness.
  - Lateral meniscus: Intact. No tear identified.
  - Joint space: Mildly reduced at 3.5 mm.

Patellofemoral compartment:
  - Grade II chondromalacia of the medial patellar facet.
  - Mild patellofemoral osteoarthritis. Patellar alta present (Insall-Salvati 1.3).
  - No patellar tendon pathology.

Soft tissues / ligaments:
  - Moderate joint effusion with estimated 15 mL fluid.
  - Mild thickening of the medial collateral ligament, no discrete tear.
  - ACL: Intact with normal signal.
  - PCL: Intact.
  - Baker's cyst: Small (2.1 cm × 1.2 cm), no rupture.

IMPRESSION:
1. Severe medial compartment osteoarthritis with Grade III chondromalacia and meniscal extrusion.
2. Horizontal posterior horn tear of the medial meniscus.
3. Moderate joint effusion.
4. Patellofemoral osteoarthritis, mild.
5. Imaging findings are consistent with the reported clinical symptoms and failed
   conservative management. Arthroscopic intervention is supported by these findings.

Electronically signed: Dr. James Patterson, MD – Radiologist (License: MD-45678)
Report finalized: June 16, 2026 at 09:42
""",
        },
        {
            "doc_id": "DOC-P001-002",
            "filename": "ortho_consult_2026_06_20.pdf",
            "type": "consultation_note",
            "pages": 3,
            "content": """
METRO ORTHOPEDIC ASSOCIATES
ORTHOPEDIC SURGERY CONSULTATION NOTE

Patient: John Doe  |  DOB: 1972-03-15  |  MRN: MRN-001234
Date of Service: June 20, 2026
Consulting Physician: Dr. Robert Chen, MD – Board Certified, Orthopedic Surgery
Referring Physician: Dr. Sarah Mitchell, MD (Primary Care)
Reason for Referral: Evaluation of right knee pain for surgical consideration.

CHIEF COMPLAINT:
Right knee pain, difficulty ascending/descending stairs, inability to walk > 2 blocks
without stopping due to pain. Pain rated 7/10 at rest, 9/10 with activity.

HISTORY OF PRESENT ILLNESS:
Mr. Doe is a 52-year-old male presenting with an 18-month history of progressively
worsening right knee pain. He initially noticed pain after prolonged standing at work
(warehouse supervisor). Pain has progressed to constant aching with sharp exacerbations.

Conservative measures completed:
  1. Physical therapy – 6 months (July 2025 – January 2026): No significant improvement.
     Therapist documented "minimal functional gains, patient reports worsening pain."
  2. Meloxicam 15 mg daily – Ongoing: Partial pain relief only (reduces pain to 6/10).
  3. Corticosteroid injections × 2:
     - October 2025: 2 weeks of moderate relief, returned to baseline.
     - February 2026: Minimal relief (< 1 week).
  4. Activity modification: Patient reports significant reduction in work and recreational activities.

REVIEW OF SYSTEMS: Positive for right knee pain, swelling, and stiffness.
Negative for fever, constitutional symptoms, or trauma.

PAST MEDICAL HISTORY: Hypertension (well-controlled).
MEDICATIONS: Meloxicam 15 mg, Lisinopril 10 mg.
ALLERGIES: Penicillin (rash).

PHYSICAL EXAMINATION:
  Gait: Antalgic, favoring right lower extremity.
  Inspection: Mild joint effusion right knee. No ecchymosis. No erythema.
  Range of motion: Flexion 95° (restricted from 135°), Extension -5° lag.
  Palpation: Medial joint line tenderness +++ (3+). Lateral joint line tenderness +.
  Special tests: McMurray's test – positive medial. Varus stress test – stable.
                 Lachman test – negative (ACL intact). Effusion confirmed (ballottement +).
  Neurovascular: Intact distally. Pulses 2+ bilateral. Sensation intact.

DIAGNOSTIC REVIEW:
MRI right knee (June 15, 2026): Grade III medial compartment chondromalacia, posterior horn
medial meniscus tear, moderate effusion – reviewed and consistent with clinical presentation.

ASSESSMENT:
  1. Severe right knee osteoarthritis – medial compartment, Grade III.
  2. Medial meniscus posterior horn tear (MRI-confirmed).
  3. Failed conservative management (18 months, all standard modalities completed).
  4. Significant functional impairment (unable to perform occupational and ADL requirements).

PLAN:
Recommend: Right knee arthroscopy with the following:
  - Partial medial meniscectomy (posterior horn tear)
  - Chondroplasty of medial femoral condyle (Grade III chondromalacia)
  - Joint debridement and lavage

Surgical details:
  - CPT: 29881 (arthroscopy with meniscectomy)
  - Facility: Outpatient surgical center
  - Estimated procedure time: 45–60 minutes
  - Expected recovery: 4–6 weeks to return to normal activities; 6–8 weeks full return to work.

Medical necessity criteria met:
  ✓ Conservative treatment > 6 months with documented failure
  ✓ MRI-confirmed structural pathology
  ✓ Significant functional impairment affecting occupational capacity

Pre-authorization requested. Patient educated on risks, benefits, and alternatives.
Written consent obtained.

Dr. Robert Chen, MD – Orthopedic Surgery  |  June 20, 2026
""",
        },
    ],

    # ── P003 – Michael Chen ──────────────────────────────────────────────────
    "P003": [
        {
            "doc_id": "DOC-P003-001",
            "filename": "oncology_treatment_plan_2026_06_01.pdf",
            "type": "consultation_note",
            "pages": 5,
            "content": """
CITY CANCER CENTER – ONCOLOGY DEPARTMENT
TREATMENT PLAN AND PROGRESS NOTE

Patient: Michael Chen  |  DOB: 1963-11-08  |  MRN: MRN-003456
Date: June 1, 2026
Oncologist: Dr. Patricia Kim, MD – Medical Oncology
Diagnosis: Non-Hodgkin Lymphoma (NHL), Diffuse Large B-Cell, Stage IIIA
ECOG Performance Status: 1

DIAGNOSIS SUMMARY:
Mr. Chen was diagnosed with Diffuse Large B-Cell Lymphoma (DLBCL) in April 2026
following evaluation of progressive cervical lymphadenopathy and constitutional symptoms
(B symptoms: night sweats, 12 lb unintentional weight loss over 3 months, fatigue).

Staging workup completed:
  - PET/CT (April 10, 2026): Hypermetabolic lymphadenopathy – bilateral cervical, mediastinal,
    retroperitoneal nodes. No organ involvement. Stage IIIA confirmed.
  - Bone marrow biopsy (April 15, 2026): Negative for lymphoma involvement.
  - Echocardiogram: LVEF 62% – adequate for anthracycline-based therapy.
  - CBC, CMP, LDH: LDH elevated 2.1× ULN (International Prognostic Index score: 2 – intermediate).

TREATMENT PROTOCOL:
R-CHOP chemotherapy (standard-of-care for DLBCL, NCCN Category 1):
  - Rituximab 375 mg/m² IV
  - Cyclophosphamide 750 mg/m² IV
  - Doxorubicin 50 mg/m² IV
  - Vincristine 1.4 mg/m² IV (max 2 mg)
  - Prednisone 100 mg PO days 1–5
  Cycle frequency: Every 21 days × 6 cycles.
  Current cycle: Cycle 3 of 6 (submitted claim for Cycle 3)

RESPONSE ASSESSMENT:
Interim PET/CT (after Cycle 2, May 25, 2026): Deauville Score 2 – Complete metabolic response.
Excellent early response. Plan to complete full 6-cycle course.

ASSESSMENT:
Patient tolerating treatment with manageable side effects (Grade 1 nausea, fatigue).
Growth factor support (Pegfilgrastim) administered. Oncology team confirms continuation of
current regimen is medically necessary and consistent with NCCN guidelines for DLBCL.

PLAN:
Cycle 3 administration per R-CHOP protocol – scheduled for June 5, 2026.
Estimated cost per cycle: $22,000–$30,000 (includes chemotherapy agents, administration,
supportive medications, labs, monitoring).

All treatment decisions consistent with NCCN Clinical Practice Guidelines in Oncology –
B-Cell Lymphomas, Version 4.2026.

Dr. Patricia Kim, MD  |  City Cancer Center  |  June 1, 2026
""",
        },
        {
            "doc_id": "DOC-P003-002",
            "filename": "lab_results_pre_cycle3_2026_06_03.pdf",
            "type": "lab_report",
            "pages": 2,
            "content": """
CITY CANCER CENTER – CLINICAL LABORATORY
PRE-CHEMOTHERAPY LAB PANEL

Patient: Michael Chen  |  MRN: MRN-003456  |  DOB: 1963-11-08
Collection Date: June 3, 2026  |  Report Date: June 3, 2026
Ordering Physician: Dr. Patricia Kim

COMPLETE BLOOD COUNT (CBC):
  WBC: 5.2 × 10³/µL  [Normal: 4.5–11.0]  ✓
  ANC: 3.1 × 10³/µL  [Normal: 1.8–7.7]   ✓  Adequate for chemotherapy
  Hemoglobin: 11.8 g/dL [Normal: 13.5–17.5] ↓ Mild anemia (stable vs. prior)
  Platelets: 187 × 10³/µL [Normal: 150–400] ✓

COMPREHENSIVE METABOLIC PANEL (CMP):
  Creatinine: 0.9 mg/dL  ✓  eGFR: 87 mL/min/1.73m²  ✓
  ALT: 28 U/L  ✓  AST: 32 U/L  ✓  Total Bilirubin: 0.8 mg/dL  ✓
  Glucose: 142 mg/dL ↑  (Patient has Type 2 Diabetes; consistent with known baseline)

LDH: 285 U/L  [Normal: 140–280] ↑ Mildly elevated, trending down (prior: 485 U/L)
  (Downtrend consistent with treatment response)

INTERPRETATION:
Laboratory values support proceeding with Cycle 3 R-CHOP as planned.
ANC adequate. Organ function preserved. LDH downtrend confirms response.
Mild anemia – monitor; transfusion threshold not met.

No laboratory abnormalities requiring treatment delay.

Electronically verified: Lab Director – Dr. Henry Park, MD  |  June 3, 2026
""",
        },
    ],

    # ── P004 – Sarah Williams ────────────────────────────────────────────────
    "P004": [
        {
            "doc_id": "DOC-P004-001",
            "filename": "lumbar_mri_2026_05_10.pdf",
            "type": "radiology_report",
            "pages": 4,
            "content": """
ADVANCED SPINE IMAGING
MRI LUMBAR SPINE WITHOUT CONTRAST

Patient: Sarah Williams  |  DOB: 1986-02-14  |  MRN: MRN-004789
Referring: Dr. James Peterson, MD  |  Exam Date: May 10, 2026

CLINICAL HISTORY:
38-year-old female with 3-year history of low back pain radiating to the left lower extremity.
Conservative treatment (PT 8 months, epidural steroid injections × 3, medications) without
adequate relief. Presenting for surgical planning.

TECHNIQUE: MRI lumbar spine 3.0T, sagittal T1/T2/STIR, axial T2 at L3-S1.

FINDINGS:
L3-L4: Mild disc desiccation. No significant stenosis. Mild bilateral facet arthropathy.
L4-L5:
  - Severe disc desiccation with loss of disc height (45% loss vs. normal).
  - Broad-based posterior disc herniation with bilateral paracentral components.
  - Central canal stenosis: Moderate (AP diameter reduced to 8 mm; normal ≥ 12 mm).
  - Bilateral lateral recess narrowing, left > right.
  - Left L5 nerve root compression: Moderate.
  - Severe facet arthropathy with ligamentum flavum hypertrophy (5 mm).
L5-S1:
  - Severe disc desiccation. Grade I spondylolisthesis (4 mm anterior translation of L5 on S1).
  - Foraminal stenosis bilateral, left > right. Left S1 nerve root compression: Mild-moderate.
  - Severe facet arthropathy. Endplate irregularity consistent with degenerative change.

Conus medullaris: Terminates at L1, normal.
Paraspinal musculature: Mild fatty atrophy bilateral, consistent with chronic pain deconditioning.

IMPRESSION:
1. Severe degenerative disc disease L4-L5 and L5-S1 with Grade I spondylolisthesis at L5-S1.
2. Moderate central canal stenosis and bilateral lateral recess narrowing at L4-L5.
3. Left L5 nerve root compression (moderate) and left S1 nerve root compression (mild-moderate).
4. Imaging findings are consistent with reported left leg radiculopathy.
5. Findings support consideration of surgical intervention given failed conservative management.

Dr. Emily Zhao, MD – Neuroradiology  |  May 10, 2026
""",
        },
        {
            "doc_id": "DOC-P004-002",
            "filename": "neurosurgery_consult_2026_05_20.pdf",
            "type": "consultation_note",
            "pages": 4,
            "content": """
REGIONAL SPINE CENTER
NEUROSURGERY CONSULTATION

Patient: Sarah Williams  |  DOB: 1986-02-14  |  MRN: MRN-004789
Date: May 20, 2026  |  Surgeon: Dr. Michael Torres, MD – Neurosurgery
Referral from: Dr. James Peterson, MD

CHIEF COMPLAINT: Chronic low back pain with left leg radiculopathy (L5, S1 distribution).
Pain 8/10 at worst, interfering with work (office administrator) and all ADLs.

HISTORY:
3-year history of progressive lumbar pain radiating into left buttock, posterior thigh,
lateral calf to dorsal foot. Constant low back ache, episodic sharp shooting pain.
Paraesthesias (numbness, tingling) in left L5/S1 dermatome.
Weakness: Left big toe extension 4/5, left ankle dorsiflexion 4/5.

Conservative treatment history:
  1. Physical therapy: 8 months (2024–2025) – minimal improvement
  2. Epidural steroid injections × 3 (last: January 2026) – temporary relief (4–6 weeks each)
  3. Gabapentin 300 mg TID – partial neuropathic pain relief
  4. Cyclobenzaprine PRN – modest muscle relaxant benefit
  5. Chiropractic: 6 months (2023) – no sustained benefit
  All standard conservative modalities exhausted without adequate symptom control.

EXAMINATION:
  Straight leg raise: Positive left at 30°.
  Motor: L4 (quad) 5/5, L5 (EHL) 4/5 (↓ left), S1 (plantar flex) 5/5.
  Sensation: Decreased left lateral calf and dorsal foot (L5 dermatome).
  Reflexes: Patellar 2+ bilateral, Achilles 1+ left (↓ vs 2+ right).

ASSESSMENT:
  1. Lumbar degenerative disc disease L4-L5, L5-S1 (MRI-confirmed).
  2. Grade I spondylolisthesis L5-S1.
  3. Left L5 and S1 radiculopathy with objective neurologic deficits.
  4. Failed conservative management (3 years, all modalities).
  5. Progressive neurological deficit (motor weakness) – surgical indication.

RECOMMENDATION:
L4-L5 and L5-S1 posterior lumbar interbody fusion (PLIF) with pedicle screw instrumentation.
Procedure addresses: nerve decompression, disc excision, and stabilization of spondylolisthesis.

CPT codes: 22630 (PLIF, L4-L5) + 22632 (additional level L5-S1) + 22842 (instrumentation)
Estimated cost: $62,000–$72,000 (facility + professional fees)
Expected recovery: 6–12 weeks to return to sedentary work; full recovery 6 months.

Surgical necessity:
  ✓ Objective neurological deficits (motor weakness) – progressive, not yet resolved
  ✓ MRI-confirmed structural pathology correlating with symptoms
  ✓ Failed 3+ years conservative treatment
  ✓ Functional impairment – unable to perform occupational duties

Pre-operative clearance requested. Patient consented and agrees to proceed.

Dr. Michael Torres, MD – Neurosurgery  |  May 20, 2026
""",
        },
        {
            "doc_id": "DOC-P004-003",
            "filename": "pt_discharge_summary_2026_04_15.pdf",
            "type": "therapy_note",
            "pages": 2,
            "content": """
METROPOLITAN PHYSICAL THERAPY
DISCHARGE SUMMARY

Patient: Sarah Williams  |  DOB: 1986-02-14  |  MRN: MRN-004789
Dates of Service: August 1, 2024 – April 15, 2025  (8 months)
Referring Provider: Dr. James Peterson, MD
Primary Diagnosis: Lumbar degenerative disc disease, L4-L5 / L5-S1; Left radiculopathy
Treating Therapist: Jessica Moore, DPT

TREATMENT PROVIDED:
Total sessions: 32 (2× per week initially, reduced to 1× per week)
Modalities: Manual therapy, therapeutic exercise, core stabilization, TENS, heat/ice,
functional movement training, ergonomic training.

GOALS & OUTCOMES:
Initial Goals         | Baseline          | Discharge         | Met?
Pain (NRS)           | 8/10              | 6/10              | Partial
Lumbar Flex          | 35°               | 45°               | Partial
Function (FOTO score)| 38 (severe limit) | 48 (mod-severe)   | Partial
Return to work       | Not working full  | Still limited     | Not met
Left leg symptoms    | Constant          | Frequent          | Not met

CLINICAL NOTE:
Ms. Williams demonstrated consistent attendance and effort throughout the 8-month program.
Despite best efforts, functional outcomes remained suboptimal, with ongoing radicular symptoms
limiting progress. Left lower extremity weakness (L5 distribution) persisted at discharge.
Patient reports the radicular symptoms are her primary limiting complaint.

RECOMMENDATION:
Patient has completed an extended course of physical therapy (8 months) without adequate
functional improvement. Further physical therapy is unlikely to provide additional benefit.
Recommend orthopedic/neurosurgery evaluation for structural management.

Discharge date: April 15, 2025 (continued Gabapentin as prescribed)

Jessica Moore, DPT  |  Metropolitan Physical Therapy  |  April 15, 2025
""",
        },
    ],

    # ── P002 – Mary Johnson (healthy / auto-approve) ────────────────────────
    "P002": [],

    # ── P005 – Robert Davis (emergency / auto-approve) ──────────────────────
    "P005": [],
}


# ══════════════════════════════════════════════════════════════════════════════
# CLAIMS HISTORY  (per patient, last 24 months)
# ══════════════════════════════════════════════════════════════════════════════

CLAIMS_HISTORY: Dict[str, List[Dict[str, Any]]] = {

    "P001": [
        {"claim_id": "H-P001-01", "date": "2024-08-12", "procedure": "Orthopedic Consult – Right Knee",
         "amount": 320.00, "status": "PAID", "flag": None},
        {"claim_id": "H-P001-02", "date": "2024-10-05", "procedure": "Corticosteroid Injection – Right Knee",
         "amount": 780.00, "status": "PAID", "flag": None},
        {"claim_id": "H-P001-03", "date": "2024-10-20", "procedure": "Physical Therapy (12 sessions)",
         "amount": 2400.00, "status": "PAID", "flag": None},
        {"claim_id": "H-P001-04", "date": "2025-02-14", "procedure": "Corticosteroid Injection – Right Knee",
         "amount": 780.00, "status": "PAID", "flag": None},
        {"claim_id": "H-P001-05", "date": "2025-06-01", "procedure": "Physical Therapy (2nd course, 12 sessions)",
         "amount": 2400.00, "status": "PAID", "flag": None},
        {"claim_id": "H-P001-06", "date": "2025-09-10", "procedure": "MRI Right Knee",
         "amount": 2200.00, "status": "PAID", "flag": None},
        {"claim_id": "H-P001-07", "date": "2025-12-15", "procedure": "Hypertension Medication (90-day)",
         "amount": 180.00, "status": "PAID", "flag": None},
        {"claim_id": "H-P001-08", "date": "2026-03-01", "procedure": "Annual Physical",
         "amount": 350.00, "status": "PAID", "flag": None},
        {"claim_id": "H-P001-09", "date": "2026-06-15", "procedure": "MRI Right Knee (pre-surgical)",
         "amount": 2200.00, "status": "PAID", "flag": None},
    ],

    "P002": [
        {"claim_id": "H-P002-01", "date": "2025-07-10", "procedure": "Annual Physical",
         "amount": 320.00, "status": "PAID", "flag": None},
    ],

    "P003": [
        {"claim_id": "H-P003-01", "date": "2026-04-15", "procedure": "Diagnostic PET/CT – Staging",
         "amount": 8500.00, "status": "PAID", "flag": None},
        {"claim_id": "H-P003-02", "date": "2026-04-20", "procedure": "Bone Marrow Biopsy",
         "amount": 3200.00, "status": "PAID", "flag": None},
        {"claim_id": "H-P003-03", "date": "2026-05-01", "procedure": "R-CHOP Chemotherapy Cycle 1",
         "amount": 26800.00, "status": "PAID", "flag": None},
        {"claim_id": "H-P003-04", "date": "2026-05-22", "procedure": "R-CHOP Chemotherapy Cycle 2",
         "amount": 25900.00, "status": "PAID", "flag": None},
        {"claim_id": "H-P003-05", "date": "2026-05-25", "procedure": "Interim PET/CT – Response Assessment",
         "amount": 8500.00, "status": "PAID", "flag": None},
    ],

    "P004": [
        {"claim_id": "H-P004-01", "date": "2024-08-01", "procedure": "Physical Therapy (8 months, 32 sessions)",
         "amount": 6400.00, "status": "PAID", "flag": None},
        {"claim_id": "H-P004-02", "date": "2024-10-15", "procedure": "Epidural Steroid Injection (1st)",
         "amount": 1850.00, "status": "PAID", "flag": None},
        {"claim_id": "H-P004-03", "date": "2025-02-10", "procedure": "Epidural Steroid Injection (2nd)",
         "amount": 1850.00, "status": "PAID", "flag": None},
        {"claim_id": "H-P004-04", "date": "2025-05-10", "procedure": "MRI Lumbar Spine",
         "amount": 2400.00, "status": "PAID", "flag": None},
        {"claim_id": "H-P004-05", "date": "2026-01-08", "procedure": "Epidural Steroid Injection (3rd)",
         "amount": 1850.00, "status": "PAID", "flag": None},
        {"claim_id": "H-P004-06", "date": "2026-05-10", "procedure": "MRI Lumbar Spine (pre-surgical)",
         "amount": 2400.00, "status": "PAID", "flag": None},
        {"claim_id": "H-P004-07", "date": "2026-05-20", "procedure": "Neurosurgery Consultation",
         "amount": 450.00, "status": "PAID", "flag": None},
    ],

    "P005": [],
}


# ══════════════════════════════════════════════════════════════════════════════
# TREATMENT GUIDELINES (per procedure code)
# ══════════════════════════════════════════════════════════════════════════════

TREATMENT_GUIDELINES: Dict[str, Dict[str, Any]] = {

    "knee_arthroscopy": {
        "procedure": "Knee Arthroscopy with Meniscectomy / Chondroplasty",
        "cpt_codes": ["29881", "29877"],
        "avg_cost_network": 8900.00,
        "avg_cost_out_of_network": 14500.00,
        "conservative_treatment_required_months": 6,
        "required_imaging": ["MRI"],
        "approval_criteria": [
            "MRI-confirmed structural pathology (meniscal tear or chondromalacia Grade III+)",
            "Conservative treatment failure: minimum 6 months",
            "Functional impairment documented",
            "No response to corticosteroid injections (if applicable)",
        ],
        "notes": "Arthroscopy for isolated osteoarthritis without structural pathology is not covered.",
    },

    "chemotherapy_rchop": {
        "procedure": "R-CHOP Chemotherapy (NHL – DLBCL)",
        "cpt_codes": ["96413", "96415", "96372", "J9310", "J9070"],
        "avg_cost_per_cycle": 24000.00,
        "standard_cycles": 6,
        "approval_criteria": [
            "Pathology-confirmed diagnosis of NHL",
            "Staging workup completed (PET/CT, bone marrow biopsy)",
            "NCCN-guideline-consistent treatment protocol",
            "Oncologist attestation of medical necessity",
            "ECOG performance status ≤ 2",
        ],
        "notes": "Each cycle requires separate prior authorization. Interim response assessment required.",
    },

    "lumbar_fusion": {
        "procedure": "Posterior Lumbar Interbody Fusion (PLIF) – 2 levels",
        "cpt_codes": ["22630", "22632", "22842"],
        "avg_cost_network": 52000.00,
        "avg_cost_out_of_network": 78000.00,
        "conservative_treatment_required_months": 6,
        "required_imaging": ["MRI", "X-ray"],
        "approval_criteria": [
            "MRI-confirmed disc herniation or stenosis causing radiculopathy",
            "Conservative treatment failure: minimum 6 months",
            "Objective neurological deficit (motor weakness, reflex changes)",
            "Neurosurgery or orthopedic spine surgeon attestation",
            "Failed epidural steroid injections (minimum 2)",
        ],
        "notes": "Multi-level fusion (>2 levels) requires additional peer-to-peer review.",
    },

    "annual_wellness": {
        "procedure": "Annual Wellness Visit / Preventive Care",
        "cpt_codes": ["G0439", "G0438", "99395"],
        "avg_cost_network": 350.00,
        "approval_criteria": [
            "Patient enrolled in plan",
            "One per calendar year",
        ],
        "notes": "Covered 100% under ACA preventive care mandate.",
    },

    "appendectomy_emergency": {
        "procedure": "Emergency Appendectomy (Laparoscopic)",
        "cpt_codes": ["44950", "44960"],
        "avg_cost_network": 9200.00,
        "approval_criteria": [
            "Emergency presentation",
            "Surgeon attestation of acute appendicitis",
        ],
        "notes": "Retrospective review only. Emergency authorization presumed.",
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# PROCEDURE CODE MAP (claim procedure → guidelines key)
# ══════════════════════════════════════════════════════════════════════════════

PROCEDURE_GUIDELINE_MAP: Dict[str, str] = {
    "Knee Arthroscopy with Meniscectomy": "knee_arthroscopy",
    "Right Knee Arthroscopy": "knee_arthroscopy",
    "Annual Wellness Examination": "annual_wellness",
    "R-CHOP Chemotherapy Cycle 3": "chemotherapy_rchop",
    "Lumbar Spinal Fusion L4-S1": "lumbar_fusion",
    "Emergency Appendectomy": "appendectomy_emergency",
}
