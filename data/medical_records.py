"""
Comprehensive mock data for the Healthcare Claims HITL Review Pipeline.

Simulates:
  - Patient profiles (patient DB) — 12 patients
  - Medical documents / PDFs (document store) — ~18 documents
  - Historical claims (claims DB) — per-patient 24-month history
  - Treatment guidelines (guidelines DB) — per procedure code

Each PDF record has verbose content so the "extract relevant sections
before LLM call" optimisation produces a meaningful token saving.
"""

from __future__ import annotations
from typing import Dict, List, Any

# ══════════════════════════════════════════════════════════════════════════════
# PATIENT PROFILES (12 patients: P001–P012)
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
        "out_of_pocket_remaining": 0.00,
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

    "P006": {
        "id": "P006",
        "name": "Lisa Wang",
        "dob": "1981-08-12",
        "age": 45,
        "gender": "Female",
        "conditions": ["Type 2 Diabetes Mellitus", "Hypertension", "Early cataracts (both eyes)"],
        "medications": ["Metformin 500 mg twice daily", "Lisinopril 20 mg daily", "Atorvastatin 40 mg daily"],
        "allergies": ["Latex"],
        "primary_care": "Dr. Michael Brown",
        "insurance_id": "INS-P006",
        "plan": "PPO Gold",
        "deductible_met": True,
        "out_of_pocket_remaining": 2000.00,
    },

    "P007": {
        "id": "P007",
        "name": "James Wilson",
        "dob": "1958-01-20",
        "age": 68,
        "gender": "Male",
        "conditions": ["Coronary Artery Disease (s/p CABG x4)", "Obstructive Sleep Apnea (suspected)", "Obesity (BMI 34)"],
        "medications": ["Aspirin 81 mg daily", "Metoprolol 50 mg twice daily", "Atorvastatin 80 mg daily", "Clopidogrel 75 mg daily"],
        "allergies": [],
        "primary_care": "Dr. Karen Taylor",
        "insurance_id": "INS-P007",
        "plan": "Medicare Advantage PPO",
        "deductible_met": True,
        "out_of_pocket_remaining": 1500.00,
    },

    "P008": {
        "id": "P008",
        "name": "Emily Rodriguez",
        "dob": "1971-05-03",
        "age": 55,
        "gender": "Female",
        "conditions": ["Severe Osteoarthritis – Right Hip", "GERD"],
        "medications": ["Celecoxib 200 mg daily", "Omeprazole 20 mg daily"],
        "allergies": ["Aspirin (hives)"],
        "primary_care": "Dr. Daniel Cruz",
        "insurance_id": "INS-P008",
        "plan": "PPO Silver",
        "deductible_met": True,
        "out_of_pocket_remaining": 3200.00,
    },

    "P009": {
        "id": "P009",
        "name": "David Kim",
        "dob": "1997-11-22",
        "age": 29,
        "gender": "Male",
        "conditions": [],
        "medications": [],
        "allergies": [],
        "primary_care": "Dr. Alice Thompson",
        "insurance_id": "INS-P009",
        "plan": "HMO Gold",
        "deductible_met": False,
        "out_of_pocket_remaining": 2500.00,
    },

    "P010": {
        "id": "P010",
        "name": "Anna Martinez",
        "dob": "1954-07-31",
        "age": 72,
        "gender": "Female",
        "conditions": ["COPD (GOLD Stage 3)", "Chronic Systolic Heart Failure (LVEF 35%)", "Hypertension", "Osteoporosis"],
        "medications": [
            "Fluticasone/Salmeterol 250/50 BID", "Tiotropium 18 mcg daily",
            "Furosemide 40 mg daily", "Lisinopril 10 mg daily",
            "Carvedilol 12.5 mg BID", "Alendronate 70 mg weekly",
        ],
        "allergies": ["Macrolide antibiotics (nausea)", "Latex"],
        "primary_care": "Dr. Robert Hayes",
        "insurance_id": "INS-P010",
        "plan": "Medicare Advantage PPO",
        "deductible_met": True,
        "out_of_pocket_remaining": 0.00,
    },

    "P011": {
        "id": "P011",
        "name": "Tom Baker",
        "dob": "1984-03-18",
        "age": 42,
        "gender": "Male",
        "conditions": ["Full-thickness rotator cuff tear – right shoulder (acute injury)", "Hyperlipidemia"],
        "medications": ["Ibuprofen 800 mg PRN", "Atorvastatin 20 mg daily"],
        "allergies": [],
        "primary_care": "Dr. Nancy Foster",
        "insurance_id": "INS-P011",
        "plan": "PPO Gold",
        "deductible_met": True,
        "out_of_pocket_remaining": 1800.00,
    },

    "P012": {
        "id": "P012",
        "name": "Rachel Green",
        "dob": "1993-06-14",
        "age": 33,
        "gender": "Female",
        "conditions": ["Pregnancy – G1P0, 28 weeks gestation"],
        "medications": ["Prenatal vitamins daily"],
        "allergies": [],
        "primary_care": "Dr. Susan Wright",
        "insurance_id": "INS-P012",
        "plan": "PPO Platinum",
        "deductible_met": True,
        "out_of_pocket_remaining": 500.00,
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# MEDICAL PDF RECORDS (per patient, approximately 18 documents total)
# ══════════════════════════════════════════════════════════════════════════════

MEDICAL_PDF_RECORDS: Dict[str, List[Dict[str, Any]]] = {

    # ── P001 – John Doe (knee) ────────────────────────────────────────────────
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
Exam: MRI Right Knee Without Contrast  |  Accession: ACC-78234

CLINICAL HISTORY:
52-year-old male with 18-month history of progressive right knee pain limiting ambulation.
Prior conservative treatment: physical therapy (6 months), NSAID therapy (Meloxicam 15 mg),
two corticosteroid injections (October 2025, February 2026). Presenting for surgical planning.

FINDINGS:
Medial compartment:
  - Grade III chondromalacia of the medial femoral condyle with full-thickness cartilage loss.
  - Medial meniscus: Horizontal tear in the posterior horn with meniscal extrusion 3.1 mm.
  - Joint space narrowing: 2.0 mm (normal 4-5 mm). Subchondral sclerosis present.
Lateral compartment: Grade II chondromalacia, lateral meniscus intact.
Patellofemoral: Grade II chondromalacia of the medial patellar facet.
Soft tissues: Moderate joint effusion (15 mL). ACL, PCL intact. Small Baker's cyst.

IMPRESSION:
1. Severe medial compartment osteoarthritis with Grade III chondromalacia and meniscal extrusion.
2. Horizontal posterior horn tear of the medial meniscus.
3. Moderate joint effusion.
4. Arthroscopic intervention is supported by these findings given failed conservative management.

Dr. James Patterson, MD – Radiologist
""",
        },
        {
            "doc_id": "DOC-P001-002",
            "filename": "ortho_consult_2026_06_20.pdf",
            "type": "consultation_note",
            "pages": 3,
            "content": """
METRO ORTHOPEDIC ASSOCIATES – ORTHOPEDIC SURGERY CONSULTATION NOTE

Patient: John Doe  |  DOB: 1972-03-15  |  MRN: MRN-001234
Consulting Physician: Dr. Robert Chen, MD – Orthopedic Surgery

CHIEF COMPLAINT: Right knee pain, difficulty with stairs, unable to walk > 2 blocks.
Pain rated 7/10 at rest, 9/10 with activity. 18-month progressive history.

Conservative measures completed:
  1. Physical therapy 6 months (July 2025 – January 2026): No significant improvement.
  2. Meloxicam 15 mg daily: Partial relief only.
  3. Corticosteroid injections x2: October 2025 (2 weeks relief), February 2026 (<1 week).
  4. Activity modification: Significant reduction in work/recreational activities.

PHYSICAL EXAMINATION:
  Gait: Antalgic. ROM: Flexion 95 deg (restricted), Extension -5 deg lag.
  Palpation: Medial joint line tenderness 3+. McMurray's test positive medial.
  Effusion confirmed (ballottement +). Neurovascular intact.

ASSESSMENT:
  1. Severe right knee osteoarthritis – medial compartment, Grade III.
  2. Medial meniscus posterior horn tear (MRI-confirmed).
  3. Failed conservative management (18 months, all modalities).
  4. Significant functional impairment.

PLAN: Right knee arthroscopy: partial medial meniscectomy, chondroplasty, joint debridement.
CPT: 29881. Outpatient. Expected recovery: 4-6 weeks.

Medical necessity criteria met: Conservative treatment >6 months with documented failure,
MRI-confirmed pathology, functional impairment affecting occupational capacity.

Dr. Robert Chen, MD  |  June 20, 2026
""",
        },
    ],

    # ── P003 – Michael Chen (oncology) ────────────────────────────────────────
    "P003": [
        {
            "doc_id": "DOC-P003-001",
            "filename": "oncology_treatment_plan_2026_06_01.pdf",
            "type": "consultation_note",
            "pages": 5,
            "content": """
CITY CANCER CENTER – ONCOLOGY DEPARTMENT – TREATMENT PLAN AND PROGRESS NOTE

Patient: Michael Chen  |  DOB: 1963-11-08  |  MRN: MRN-003456
Oncologist: Dr. Patricia Kim, MD – Medical Oncology
Diagnosis: Non-Hodgkin Lymphoma (NHL), Diffuse Large B-Cell, Stage IIIA
ECOG Performance Status: 1

DIAGNOSIS SUMMARY:
Diagnosed with DLBCL in April 2026 following progressive cervical lymphadenopathy and
constitutional symptoms (B symptoms: night sweats, 12 lb weight loss, fatigue).
Staging: PET/CT – hypermetabolic lymphadenopathy cervical, mediastinal, retroperitoneal.
Bone marrow biopsy negative. LVEF 62%. IPI score 2 (intermediate).

TREATMENT PROTOCOL: R-CHOP chemotherapy (NCCN Category 1):
Rituximab 375 mg/m2, Cyclophosphamide 750 mg/m2, Doxorubicin 50 mg/m2,
Vincristine 1.4 mg/m2, Prednisone 100 mg PO days 1-5. Every 21 days x 6 cycles.

RESPONSE: Interim PET/CT after Cycle 2: Deauville Score 2 – Complete metabolic response.

ASSESSMENT: Patient tolerating treatment with manageable side effects. Growth factor
support administered. Continuation of current regimen is medically necessary.

Dr. Patricia Kim, MD  |  June 1, 2026
""",
        },
        {
            "doc_id": "DOC-P003-002",
            "filename": "lab_results_pre_cycle3_2026_06_03.pdf",
            "type": "lab_report",
            "pages": 2,
            "content": """
CITY CANCER CENTER – CLINICAL LABORATORY – PRE-CHEMOTHERAPY LAB PANEL

Patient: Michael Chen  |  MRN: MRN-003456  |  DOB: 1963-11-08
Collection: June 3, 2026  |  Ordering: Dr. Patricia Kim

COMPLETE BLOOD COUNT (CBC):
  WBC: 5.2 x10^3/uL [Normal]   ANC: 3.1 x10^3/uL [Adequate for chemotherapy]
  Hemoglobin: 11.8 g/dL [Mild anemia, stable]   Platelets: 187 x10^3/uL [Normal]

COMPREHENSIVE METABOLIC PANEL (CMP):
  Creatinine: 0.9 mg/dL   eGFR: 87 mL/min/1.73m2 [Normal]
  ALT: 28 U/L   AST: 32 U/L   Bilirubin: 0.8 mg/dL [All normal]
  Glucose: 142 mg/dL Elevated (known T2DM, consistent with baseline)

LDH: 285 U/L Mildly elevated, trending down (prior: 485 U/L) – consistent with response.

INTERPRETATION:
Laboratory values support proceeding with Cycle 3 R-CHOP as planned. ANC adequate.
Organ function preserved. LDH downtrend confirms response. No lab abnormalities
requiring treatment delay.

Dr. Henry Park, MD – Lab Director  |  June 3, 2026
""",
        },
    ],

    # ── P004 – Sarah Williams (spine) ─────────────────────────────────────────
    "P004": [
        {
            "doc_id": "DOC-P004-001",
            "filename": "lumbar_mri_2026_05_10.pdf",
            "type": "radiology_report",
            "pages": 4,
            "content": """
ADVANCED SPINE IMAGING – MRI LUMBAR SPINE WITHOUT CONTRAST

Patient: Sarah Williams  |  DOB: 1986-02-14  |  MRN: MRN-004789
Referring: Dr. James Peterson, MD  |  Exam Date: May 10, 2026

CLINICAL HISTORY: 38-year-old female with 3-year history of low back pain radiating to
left lower extremity. Conservative treatment (PT 8 months, epidural steroid injections x3,
medications) without adequate relief. Presenting for surgical planning.

FINDINGS:
L4-L5: Severe disc desiccation with 45% disc height loss. Broad-based posterior disc
herniation. Central canal stenosis: Moderate (AP diameter 8 mm; normal >=12 mm).
Bilateral lateral recess narrowing, left > right. Left L5 nerve root compression: Moderate.
Severe facet arthropathy with ligamentum flavum hypertrophy (5 mm).
L5-S1: Severe disc desiccation. Grade I spondylolisthesis (4 mm anterior translation).
Foraminal stenosis bilateral, left > right. Left S1 nerve root compression: Mild-moderate.
Severe facet arthropathy with endplate irregularity.

IMPRESSION:
1. Severe DDD L4-L5 and L5-S1 with Grade I spondylolisthesis at L5-S1.
2. Moderate central canal stenosis at L4-L5.
3. Left L5 and S1 nerve root compression.
4. Findings support surgical intervention given failed conservative management.

Dr. Emily Zhao, MD – Neuroradiology  |  May 10, 2026
""",
        },
        {
            "doc_id": "DOC-P004-002",
            "filename": "neurosurgery_consult_2026_05_20.pdf",
            "type": "consultation_note",
            "pages": 4,
            "content": """
REGIONAL SPINE CENTER – NEUROSURGERY CONSULTATION

Patient: Sarah Williams  |  DOB: 1986-02-14  |  MRN: MRN-004789
Surgeon: Dr. Michael Torres, MD – Neurosurgery  |  Date: May 20, 2026

CHIEF COMPLAINT: Chronic low back pain with left leg radiculopathy (L5, S1 distribution).
Pain 8/10 at worst. Weakness: Left big toe extension 4/5, left ankle dorsiflexion 4/5.

HISTORY: 3-year progressive lumbar pain. Conservative: PT 8 months (minimal improvement),
epidural steroid injections x3 (temporary relief 4-6 weeks), Gabapentin 300 mg TID,
Cyclobenzaprine PRN. All modalities exhausted without adequate symptom control.

EXAMINATION: SLR positive left at 30 deg. Motor: L4 5/5, L5 (EHL) 4/5 decreased left,
S1 5/5. Sensation decreased left L5 dermatome. Reflex: Achilles 1+ left (decreased vs 2+ right).

ASSESSMENT:
  1. Lumbar DDD L4-L5, L5-S1 with Grade I spondylolisthesis.
  2. Left L5 and S1 radiculopathy with objective neurologic deficits (motor weakness).
  3. Failed conservative management (3 years, all modalities).
  4. Progressive neurologic deficit – surgical indication.

RECOMMENDATION: L4-L5 and L5-S1 PLIF with pedicle screw instrumentation.
CPT: 22630+22632+22842. Estimated cost: $62,000-$72,000.

Dr. Michael Torres, MD  |  May 20, 2026
""",
        },
        {
            "doc_id": "DOC-P004-003",
            "filename": "pt_discharge_summary_2026_04_15.pdf",
            "type": "therapy_note",
            "pages": 2,
            "content": """
METROPOLITAN PHYSICAL THERAPY – DISCHARGE SUMMARY

Patient: Sarah Williams  |  DOB: 1986-02-14  |  MRN: MRN-004789
Dates: August 1, 2024 – April 15, 2025 (8 months)
Therapist: Jessica Moore, DPT

TREATMENT: 32 sessions total. Modalities: manual therapy, exercise, core stabilization,
TENS, ergonomic training.

OUTCOMES:
  Pain (NRS): Baseline 8/10 -> Discharge 6/10 (Partial improvement only)
  Lumbar Flex: 35 deg -> 45 deg (Partial)
  Function (FOTO): 38 (severe limit) -> 48 (mod-severe) (Partial)
  Return to work: Not met. Left leg symptoms: Constant -> Frequent (Not met)

CLINICAL NOTE: Despite consistent attendance and effort over 8 months, functional
outcomes remained suboptimal. Left L5 weakness persisted. Further PT unlikely to
provide additional benefit. Recommend neurosurgery evaluation.

Jessica Moore, DPT  |  April 15, 2025
""",
        },
    ],

    # ── P006 – Lisa Wang (diabetes, foot ulcer, cataract) ─────────────────────
    "P006": [
        {
            "doc_id": "DOC-P006-001",
            "filename": "endocrinology_visit_2026_07_10.pdf",
            "type": "consultation_note",
            "pages": 3,
            "content": """
ENDOCRINOLOGY & DIABETES CENTER – DIABETES MANAGEMENT VISIT

Patient: Lisa Wang  |  DOB: 1981-08-12  |  MRN: MRN-006123
Endocrinologist: Dr. Amanda Foster, MD  |  Date: July 10, 2026

CHIEF COMPLAINT: Poorly controlled diabetes – HbA1c 8.4% (up from 7.2% six months ago).
Patient reports difficulty adhering to diet plan, missed medication doses, 8 lb weight gain.

MEDICATIONS: Metformin 500 mg BID, Lisinopril 20 mg daily, Atorvastatin 40 mg daily.

EXAMINATION: BP 148/92 mmHg. BMI 31.4. Foot exam: Diminished monofilament sensation
left plantar surface (2/3 sites not felt). Mild callus left heel.

LABS: HbA1c 8.4%, Fasting glucose 172 mg/dL, eGFR 72 (CKD Stage 2).

ASSESSMENT:
  1. Type 2 Diabetes – inadequately controlled (HbA1c 8.4%, target < 7.0%).
  2. Hypertension – uncontrolled.
  3. Early diabetic neuropathy – left foot sensory deficit.

PLAN: Increase Metformin to 1000 mg BID. Add Empagliflozin 10 mg daily. Podiatry referral
for diabetic foot care. Ophthalmology referral for diabetic eye exam (overdue 8 months).
Follow-up 3 months.

Dr. Amanda Foster, MD  |  July 10, 2026
""",
        },
        {
            "doc_id": "DOC-P006-002",
            "filename": "wound_care_consult_2026_08_05.pdf",
            "type": "consultation_note",
            "pages": 3,
            "content": """
WOUND CARE CENTER – DIABETIC FOOT ULCER EVALUATION AND TREATMENT

Patient: Lisa Wang  |  DOB: 1981-08-12  |  MRN: MRN-006123
Attending: Dr. Mark Stevens, DPM – Podiatry  |  Date: August 5, 2026

CHIEF COMPLAINT: Painless ulcer on left plantar foot – noted 10 days ago, not improving.

EXAMINATION: Left plantar foot – 1.8 cm x 1.2 cm ulcer at 2nd metatarsal head.
Depth: partial thickness, Wagner Grade 1. No purulence, no odor, mild peri-wound erythema.
Probe-to-bone: Negative. Dorsalis pedis: 1+ (diminished but present).
Pedal X-ray: No osteomyelitis, no foreign body.

ASSESSMENT:
  1. Diabetic neuropathic plantar foot ulcer – Wagner Grade 1, left foot.
  2. No evidence of osteomyelitis or deep infection.
  3. Contributing: sensory neuropathy, poorly controlled diabetes, callus formation.

PLAN: Sharp debridement (performed today). Hydrocolloid dressing with offloading pad.
Total contact cast for pressure offloading. Cephalexin 500 mg QID x 7 days prophylactic.
Weekly wound checks x 4 weeks. Diabetes management intensification coordinated with endocrinology.

Dr. Mark Stevens, DPM  |  August 5, 2026
""",
        },
    ],

    # ── P007 – James Wilson (cardiac, sleep) ──────────────────────────────────
    "P007": [
        {
            "doc_id": "DOC-P007-001",
            "filename": "cardiology_followup_2026_07_12.pdf",
            "type": "consultation_note",
            "pages": 3,
            "content": """
CARDIAC CARE ASSOCIATES – POST-CABG CARDIOLOGY FOLLOW-UP

Patient: James Wilson  |  DOB: 1958-01-20  |  MRN: MRN-007456
Cardiologist: Dr. Richard Lee, MD  |  Date: July 12, 2026

INTERVAL HISTORY: 68-year-old male, 4 months post-CABG x4 (LIMA-LAD, SVG-OM1, SVG-PDA,
SVG-RCA) performed March 8, 2026. Reports mild exertional dyspnea when walking >1 block.
No chest pain. Loud snoring and witnessed apneas per wife. Daytime sleepiness
(Epworth Sleepiness Scale: 14/24 – abnormal). Completed cardiac rehab Phase II.

MEDICATIONS: Aspirin 81 mg, Metoprolol 50 mg BID, Atorvastatin 80 mg, Clopidogrel 75 mg.

EXAMINATION: BP 128/78, HR 68 regular, BMI 34.2. Chest: Sternotomy healed.
Heart: Regular, no murmurs. Lungs: Clear. Extremities: No edema.

EKG: Normal sinus rhythm, rate 68 bpm. No ST-T changes.

ASSESSMENT:
  1. CAD post-CABG x4 – clinically stable. Mild exertional dyspnea warrants stress echo.
  2. Suspected Obstructive Sleep Apnea: Elevated ESS, witnessed apneas, obesity.
     Recommend polysomnography.
  3. Medications appropriate. Continue DAPT per protocol.

PLAN: Schedule dobutamine stress echo. Refer to Sleep Disorders Center for PSG with CPAP.
Continue medications. Follow-up 3 months.

Dr. Richard Lee, MD  |  July 12, 2026
""",
        },
        {
            "doc_id": "DOC-P007-002",
            "filename": "stress_echo_2026_07_25.pdf",
            "type": "radiology_report",
            "pages": 3,
            "content": """
CARDIAC IMAGING CENTER – DOBUTAMINE STRESS ECHOCARDIOGRAM

Patient: James Wilson  |  DOB: 1958-01-20  |  MRN: MRN-007456
Interpreting: Dr. Maria Santos, MD – Cardiology  |  Date: July 25, 2026

INDICATION: Post-CABG evaluation of exertional dyspnea. Rule out graft stenosis/occlusion.

PROTOCOL: Dobutamine incremental doses. Target HR 131 bpm achieved. No chest pain.

FINDINGS:
Resting: LVEF 48% (mildly reduced). Inferior/posterior wall mild hypokinesis (baseline).
No significant valvular abnormalities. PASP 22 mmHg (normal).
Peak stress (HR 131): LVEF 58% (improved – good contractile reserve). Inferior wall
normalizes at peak. No new wall motion abnormalities. GLS: -16.2% (rest) to -19.8% (peak).
Right ventricle: Normal size and function at peak.

IMPRESSION:
1. No evidence of ischemia. Good contractile reserve.
2. Resting LVEF 48% improving to 58% at peak – consistent with post-CABG stunning.
3. No evidence of graft failure. Grafts appear patent.
4. Exertional dyspnea likely multifactorial: deconditioning, obesity, possible OSA.

Dr. Maria Santos, MD  |  July 25, 2026
""",
        },
    ],

    # ── P008 – Emily Rodriguez (hip) ──────────────────────────────────────────
    "P008": [
        {
            "doc_id": "DOC-P008-001",
            "filename": "hip_xray_and_consult_2026_07_01.pdf",
            "type": "consultation_note",
            "pages": 4,
            "content": """
ADVANCED ORTHOPEDICS CENTER – HIP REPLACEMENT EVALUATION

Patient: Emily Rodriguez  |  DOB: 1971-05-03  |  MRN: MRN-008789
Consulting Surgeon: Dr. Kevin Park, MD – Joint Replacement  |  Date: July 1, 2026

CHIEF COMPLAINT: Right hip pain x 4 years, progressively worsening. Pain 7/10 constant,
8/10 with weight-bearing. Cane-dependent. Sleep disturbed by pain.

HISTORY: 4-year history of right hip OA. Conservative: PT 4 months (minimal benefit),
Celecoxib 200 mg daily (partial relief), corticosteroid injection x2 (last: July 2025,
3 weeks relief). Activity level significantly reduced.

EXAMINATION: Antalgic gait with cane. Trendelenburg sign positive right.
Hip ROM: Flexion 75 deg (normal 120), internal rotation 5 deg (normal 30).
Leg length: Apparent shortening right (1.2 cm).

IMAGING: Hip X-ray (June 28, 2026): Severe joint space narrowing (0.5 mm), subchondral
cysts, marginal osteophytes, femoral head flattening. Kellgren-Lawrence Grade IV.

ASSESSMENT:
  1. End-stage osteoarthritis – right hip (K-L Grade IV).
  2. Failed conservative management (4 years, all modalities).
  3. Significant functional impairment. Surgical candidate for THA.

PLAN: Total hip replacement – right (cementless, posterior approach). CPT: 27130.
Estimated cost: $38,000-$45,000. Outpatient procedure.

Dr. Kevin Park, MD  |  July 1, 2026
""",
        },
    ],

    # ── P010 – Anna Martinez (COPD, CHF) ──────────────────────────────────────
    "P010": [
        {
            "doc_id": "DOC-P010-001",
            "filename": "copd_admission_hp_2026_07_20.pdf",
            "type": "consultation_note",
            "pages": 4,
            "content": """
CITY GENERAL HOSPITAL – INPATIENT ADMISSION HISTORY AND PHYSICAL

Patient: Anna Martinez  |  DOB: 1954-07-31  |  MRN: MRN-010321
Admitting: Dr. Robert Hayes, MD – Hospitalist  |  Admission: July 20, 2026

CHIEF COMPLAINT: Acute shortness of breath x 2 days, productive cough, wheezing.

HISTORY: 72-year-old female with COPD (GOLD Stage 3, home O2 2L NC PRN) and CHF
(LVEF 35%) presents with 2-day history of increased dyspnea, purulent sputum, wheezing.
O2 sat 84% on room air (improved to 91% on 3L NC). 3 prior exacerbations in 12 months
(2 requiring hospitalization). Ran out of inhalers 5 days ago (medication non-adherence).

EXAMINATION: Tachypneic (RR 28), accessory muscle use. Lungs: Diffuse expiratory
wheezing, coarse rhonchi at bases. Heart: Tachycardic (HR 104), regular, no S3.
Extremities: 1+ pitting edema bilaterally. Clubbing +.

LABS/IMAGING: ABG on 3L NC: pH 7.34, pCO2 52, pO2 68, HCO3 28.
WBC 14.2 (left shift). BNP 620. CXR: Hyperinflated lungs, no infiltrate.
EKG: Sinus tachycardia, no acute changes.

ASSESSMENT:
  1. Acute COPD exacerbation – infectious trigger suspected.
  2. Acute-on-chronic respiratory failure (CO2 retention).
  3. CHF – compensated. Medication non-adherence (inhalers).

PLAN: Admit to medicine floor. O2 to keep SpO2 88-92%. IV Methylprednisolone 40 mg q8h
x 3 days then taper. Nebulized Ipratropium/Albuterol q4h. Azithromycin 500 mg IV x 5 days.
Resume home medications. Cardiology consult for CHF optimization.

Dr. Robert Hayes, MD  |  July 20, 2026
""",
        },
        {
            "doc_id": "DOC-P010-002",
            "filename": "echo_tt_2026_08_10.pdf",
            "type": "radiology_report",
            "pages": 2,
            "content": """
CARDIAC IMAGING CENTER – TRANSTHORACIC ECHOCARDIOGRAM

Patient: Anna Martinez  |  DOB: 1954-07-31  |  MRN: MRN-010321
Interpreting: Dr. Maria Santos, MD – Cardiology  |  Date: August 10, 2026

INDICATION: Post-COPD exacerbation reassessment of LV function and pulmonary pressures.

FINDINGS:
  LV size: Normal. LVEF: 35% (unchanged, severely reduced).
  Regional wall motion: Inferior and inferoseptal hypokinesis.
  RV size: Mildly enlarged. RV function: Mildly reduced (TAPSE 14 mm).
  Estimated PASP: 42 mmHg (mildly elevated – mild pulmonary hypertension).
  Valves: Mild mitral regurgitation. Trace tricuspid regurgitation.
  Pericardium: No effusion.

IMPRESSION:
  1. Severely reduced LV systolic function (LVEF 35%) – unchanged, chronic.
  2. Mildly elevated pulmonary pressures (PASP 42 mmHg) – likely Group 3 (COPD-related)
     with possible Group 2 (left heart) contribution.
  3. No acute changes from prior. Continue GDMT.

Dr. Maria Santos, MD  |  August 10, 2026
""",
        },
    ],

    # ── P011 – Tom Baker (shoulder) ───────────────────────────────────────────
    "P011": [
        {
            "doc_id": "DOC-P011-001",
            "filename": "shoulder_mri_2026_07_22.pdf",
            "type": "radiology_report",
            "pages": 3,
            "content": """
SPORTS MEDICINE IMAGING – MRI SHOULDER RIGHT WITHOUT CONTRAST

Patient: Tom Baker  |  DOB: 1984-03-18  |  MRN: MRN-011654
Referring: Dr. Nancy Foster, MD  |  Date: July 22, 2026

CLINICAL HISTORY: 42-year-old male with acute right shoulder injury 4 weeks ago
(fall while rock climbing, arm extended overhead). Persistent pain, weakness with
overhead activities and external rotation. Night pain. Failed conservative management
(rest, NSAIDs, PT x 3 weeks).

FINDINGS:
Supraspinatus tendon: Full-thickness tear involving anterior 2.0 cm of insertion at
greater tuberosity. Tear measures 2.5 cm AP x 1.8 cm ML. Tendon retraction: 2.8 cm
(Patte Grade 2 – moderate). Fatty infiltration: Goutallier Grade 1 (mild).
Infraspinatus: Intact. Subscapularis: Intact. Biceps tendon: Mild tendinosis, intact.
Labrum: Degenerative fraying superior labrum (SLAP Type I).
AC joint: Mild osteoarthritis. Subacromial space: 7 mm (narrowed).
Joint: Moderate glenohumeral joint effusion. No loose bodies.

IMPRESSION:
1. Full-thickness supraspinatus tendon tear (2.5 cm AP with 2.8 cm retraction).
2. Moderate subacromial impingement. AC joint OA mild.
3. No other rotator cuff tendon tears.
4. Surgical consultation recommended – full-thickness tear, acute mechanism, age 42,
   high functional demands.

Dr. David Morgan, MD – Musculoskeletal Radiology  |  July 22, 2026
""",
        },
        {
            "doc_id": "DOC-P011-002",
            "filename": "shoulder_surgery_consult_2026_08_01.pdf",
            "type": "consultation_note",
            "pages": 3,
            "content": """
SPORTS MEDICINE SURGERY CENTER – SHOULDER SURGERY CONSULTATION

Patient: Tom Baker  |  DOB: 1984-03-18  |  MRN: MRN-011654
Surgeon: Dr. Erik Johansson, MD – Sports Medicine  |  Date: August 1, 2026

CHIEF COMPLAINT: Right shoulder pain and weakness since acute injury 5 weeks ago.
Pain 6/10 at rest, 9/10 with overhead reach or external rotation. Night pain present.

HISTORY: Acute injury while rock climbing (fell, arm extended overhead). Treated with
3 weeks PT + NSAIDs without significant improvement. MRI performed July 22, 2026.

EXAMINATION: Painful arc 60-120 deg. ROM: Forward elevation 120 deg decreased,
External rotation 30 deg decreased. Drop arm test positive at 60 deg.
Jobe (empty can) test: positive (pain + weakness). Hawkins-Kennedy: positive.
Strength: Supraspinatus 3/5 (significant weakness). Infraspinatus 5/5, Subscapularis 5/5.

ASSESSMENT:
  1. Full-thickness supraspinatus tendon tear (MRI-confirmed, 2.5 cm, retracted 2.8 cm).
  2. Failed conservative management (PT + NSAIDs, 5 weeks).
  3. Significant functional impairment affecting work (carpenter) and recreation.
  4. Surgical candidate for arthroscopic rotator cuff repair.

PLAN: Arthroscopic double-row suture bridge rotator cuff repair + subacromial decompression.
CPT: 23427 + 29826. Estimated cost: $17,000-$22,000.

Dr. Erik Johansson, MD  |  August 1, 2026
""",
        },
    ],

    # ── Patients without documents ─────────────────────────────────────────────
    "P002": [],
    "P005": [],
    "P009": [],
    "P012": [],
}


# ══════════════════════════════════════════════════════════════════════════════
# CLAIMS HISTORY (per patient, last 24 months)
# ══════════════════════════════════════════════════════════════════════════════

CLAIMS_HISTORY: Dict[str, List[Dict[str, Any]]] = {

    "P001": [
        {"claim_id": "H-P001-01", "date": "2024-08-12", "procedure": "Orthopedic Consult – Right Knee", "amount": 320.00, "status": "PAID", "flag": None},
        {"claim_id": "H-P001-02", "date": "2024-10-05", "procedure": "Corticosteroid Injection – Right Knee", "amount": 780.00, "status": "PAID", "flag": None},
        {"claim_id": "H-P001-03", "date": "2024-10-20", "procedure": "Physical Therapy (12 sessions)", "amount": 2400.00, "status": "PAID", "flag": None},
        {"claim_id": "H-P001-04", "date": "2025-02-14", "procedure": "Corticosteroid Injection – Right Knee", "amount": 780.00, "status": "PAID", "flag": None},
        {"claim_id": "H-P001-05", "date": "2025-06-01", "procedure": "Physical Therapy (2nd course, 12 sessions)", "amount": 2400.00, "status": "PAID", "flag": None},
        {"claim_id": "H-P001-06", "date": "2025-09-10", "procedure": "MRI Right Knee", "amount": 2200.00, "status": "PAID", "flag": None},
        {"claim_id": "H-P001-07", "date": "2025-12-15", "procedure": "Hypertension Medication (90-day)", "amount": 180.00, "status": "PAID", "flag": None},
        {"claim_id": "H-P001-08", "date": "2026-03-01", "procedure": "Annual Physical", "amount": 350.00, "status": "PAID", "flag": None},
        {"claim_id": "H-P001-09", "date": "2026-06-15", "procedure": "MRI Right Knee (pre-surgical)", "amount": 2200.00, "status": "PAID", "flag": None},
    ],

    "P002": [
        {"claim_id": "H-P002-01", "date": "2025-07-10", "procedure": "Annual Physical", "amount": 320.00, "status": "PAID", "flag": None},
        {"claim_id": "H-P002-02", "date": "2026-01-15", "procedure": "Dermatology Consult", "amount": 280.00, "status": "PAID", "flag": None},
    ],

    "P003": [
        {"claim_id": "H-P003-01", "date": "2026-04-15", "procedure": "Diagnostic PET/CT – Staging", "amount": 8500.00, "status": "PAID", "flag": None},
        {"claim_id": "H-P003-02", "date": "2026-04-20", "procedure": "Bone Marrow Biopsy", "amount": 3200.00, "status": "PAID", "flag": None},
        {"claim_id": "H-P003-03", "date": "2026-05-01", "procedure": "R-CHOP Chemotherapy Cycle 1", "amount": 26800.00, "status": "PAID", "flag": None},
        {"claim_id": "H-P003-04", "date": "2026-05-22", "procedure": "R-CHOP Chemotherapy Cycle 2", "amount": 25900.00, "status": "PAID", "flag": None},
        {"claim_id": "H-P003-05", "date": "2026-05-25", "procedure": "Interim PET/CT – Response Assessment", "amount": 8500.00, "status": "PAID", "flag": None},
    ],

    "P004": [
        {"claim_id": "H-P004-01", "date": "2024-08-01", "procedure": "Physical Therapy (8 months, 32 sessions)", "amount": 6400.00, "status": "PAID", "flag": None},
        {"claim_id": "H-P004-02", "date": "2024-10-15", "procedure": "Epidural Steroid Injection (1st)", "amount": 1850.00, "status": "PAID", "flag": None},
        {"claim_id": "H-P004-03", "date": "2025-02-10", "procedure": "Epidural Steroid Injection (2nd)", "amount": 1850.00, "status": "PAID", "flag": None},
        {"claim_id": "H-P004-04", "date": "2025-05-10", "procedure": "MRI Lumbar Spine", "amount": 2400.00, "status": "PAID", "flag": None},
        {"claim_id": "H-P004-05", "date": "2026-01-08", "procedure": "Epidural Steroid Injection (3rd)", "amount": 1850.00, "status": "PAID", "flag": None},
        {"claim_id": "H-P004-06", "date": "2026-05-10", "procedure": "MRI Lumbar Spine (pre-surgical)", "amount": 2400.00, "status": "PAID", "flag": None},
        {"claim_id": "H-P004-07", "date": "2026-05-20", "procedure": "Neurosurgery Consultation", "amount": 450.00, "status": "PAID", "flag": None},
    ],

    "P005": [
        {"claim_id": "H-P005-01", "date": "2025-09-20", "procedure": "ER Visit – Laceration Repair", "amount": 1800.00, "status": "PAID", "flag": None},
    ],

    "P006": [
        {"claim_id": "H-P006-01", "date": "2025-08-10", "procedure": "Endocrinology Visit", "amount": 480.00, "status": "PAID", "flag": None},
        {"claim_id": "H-P006-02", "date": "2025-11-05", "procedure": "HbA1c + Lipid Panel", "amount": 150.00, "status": "PAID", "flag": None},
        {"claim_id": "H-P006-03", "date": "2026-01-15", "procedure": "Diabetes Management Visit", "amount": 490.00, "status": "PAID", "flag": None},
        {"claim_id": "H-P006-04", "date": "2026-03-10", "procedure": "Retinal Eye Exam", "amount": 250.00, "status": "PAID", "flag": None},
        {"claim_id": "H-P006-05", "date": "2026-05-20", "procedure": "Metformin + Lisinopril (90-day)", "amount": 95.00, "status": "PAID", "flag": None},
    ],

    "P007": [
        {"claim_id": "H-P007-01", "date": "2026-03-01", "procedure": "CABG x4 – Inpatient", "amount": 128000.00, "status": "PAID", "flag": None},
        {"claim_id": "H-P007-02", "date": "2026-03-20", "procedure": "Post-CABG Hospital Follow-Up", "amount": 380.00, "status": "PAID", "flag": None},
        {"claim_id": "H-P007-03", "date": "2026-04-15", "procedure": "Cardiac Rehab Phase I – Inpatient", "amount": 2400.00, "status": "PAID", "flag": None},
        {"claim_id": "H-P007-04", "date": "2026-05-01", "procedure": "Cardiac Rehab Phase II (12 sessions)", "amount": 3200.00, "status": "PAID", "flag": None},
        {"claim_id": "H-P007-05", "date": "2026-06-10", "procedure": "Lipid Panel + BMP", "amount": 140.00, "status": "PAID", "flag": None},
    ],

    "P008": [
        {"claim_id": "H-P008-01", "date": "2025-01-20", "procedure": "Hip X-Ray (right)", "amount": 450.00, "status": "PAID", "flag": None},
        {"claim_id": "H-P008-02", "date": "2025-06-15", "procedure": "Physical Therapy (4 months)", "amount": 3200.00, "status": "PAID", "flag": None},
        {"claim_id": "H-P008-03", "date": "2025-07-01", "procedure": "Corticosteroid Injection – Right Hip", "amount": 650.00, "status": "PAID", "flag": None},
        {"claim_id": "H-P008-04", "date": "2026-03-10", "procedure": "Corticosteroid Injection – Right Hip (2nd)", "amount": 650.00, "status": "PAID", "flag": None},
        {"claim_id": "H-P008-05", "date": "2026-06-28", "procedure": "Hip X-Ray (pre-surgical)", "amount": 450.00, "status": "PAID", "flag": None},
    ],

    "P009": [
        {"claim_id": "H-P009-01", "date": "2025-07-01", "procedure": "Annual Physical", "amount": 320.00, "status": "PAID", "flag": None},
    ],

    "P010": [
        {"claim_id": "H-P010-01", "date": "2025-08-05", "procedure": "COPD Exacerbation – Hospitalization (4 days)", "amount": 22500.00, "status": "PAID", "flag": None},
        {"claim_id": "H-P010-02", "date": "2025-10-20", "procedure": "Cardiology Follow-Up", "amount": 420.00, "status": "PAID", "flag": None},
        {"claim_id": "H-P010-03", "date": "2025-12-10", "procedure": "Pulmonary Function Tests", "amount": 650.00, "status": "PAID", "flag": None},
        {"claim_id": "H-P010-04", "date": "2026-02-01", "procedure": "COPD Exacerbation – Hospitalization (3 days)", "amount": 19500.00, "status": "PAID", "flag": None},
        {"claim_id": "H-P010-05", "date": "2026-03-15", "procedure": "Echocardiogram", "amount": 2100.00, "status": "PAID", "flag": None},
        {"claim_id": "H-P010-06", "date": "2026-05-10", "procedure": "Pulmonology Follow-Up", "amount": 380.00, "status": "PAID", "flag": None},
    ],

    "P011": [
        {"claim_id": "H-P011-01", "date": "2025-04-01", "procedure": "Annual Physical", "amount": 320.00, "status": "PAID", "flag": None},
        {"claim_id": "H-P011-02", "date": "2025-09-15", "procedure": "Lipid Panel + BMP", "amount": 140.00, "status": "PAID", "flag": None},
        {"claim_id": "H-P011-03", "date": "2026-07-01", "procedure": "Shoulder X-Ray (right)", "amount": 380.00, "status": "PAID", "flag": None},
    ],

    "P012": [
        {"claim_id": "H-P012-01", "date": "2026-03-15", "procedure": "Prenatal Visit – 12 weeks (initial OB)", "amount": 680.00, "status": "PAID", "flag": None},
        {"claim_id": "H-P012-02", "date": "2026-05-10", "procedure": "Prenatal Visit – 20 weeks (anatomy scan)", "amount": 520.00, "status": "PAID", "flag": None},
        {"claim_id": "H-P012-03", "date": "2026-06-20", "procedure": "Prenatal Visit – 24 weeks", "amount": 480.00, "status": "PAID", "flag": None},
    ],
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
            "ECOG performance status <= 2",
        ],
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
    },

    "annual_wellness": {
        "procedure": "Annual Wellness Visit / Preventive Care",
        "cpt_codes": ["G0439", "G0438", "99395"],
        "avg_cost_network": 350.00,
        "approval_criteria": ["Patient enrolled in plan", "One per calendar year"],
    },

    "appendectomy_emergency": {
        "procedure": "Emergency Appendectomy (Laparoscopic)",
        "cpt_codes": ["44950", "44960"],
        "avg_cost_network": 9200.00,
        "approval_criteria": ["Emergency presentation", "Surgeon attestation of acute appendicitis"],
    },

    "pt_knee_rehab": {
        "procedure": "Post-Operative Physical Therapy – Knee",
        "cpt_codes": ["97110", "97112", "97140", "97530"],
        "avg_cost_network": 2400.00,
        "approval_criteria": [
            "Status post-knee surgery (within 3 months)",
            "Physician prescription with documented functional deficits",
        ],
    },

    "diabetes_management": {
        "procedure": "Diabetes Management Comprehensive Visit",
        "cpt_codes": ["99214", "G0108", "G0109"],
        "avg_cost_network": 480.00,
        "approval_criteria": [
            "Documented diabetes diagnosis",
            "HbA1c testing within last 3 months",
            "Annual foot and eye exam recommended",
        ],
    },

    "wound_care_diabetic": {
        "procedure": "Diabetic Foot Ulcer Debridement and Wound Care",
        "cpt_codes": ["11042", "97597", "97598"],
        "avg_cost_network": 2800.00,
        "approval_criteria": [
            "Documented diabetic neuropathy",
            "Plantar ulcer – Wagner Grade 1 or 2",
            "Offloading device prescribed",
            "Weekly wound assessments required",
        ],
    },

    "hip_replacement": {
        "procedure": "Total Hip Arthroplasty – Cementless",
        "cpt_codes": ["27130"],
        "avg_cost_network": 34000.00,
        "avg_cost_out_of_network": 48000.00,
        "conservative_treatment_required_months": 6,
        "required_imaging": ["X-ray"],
        "approval_criteria": [
            "Radiographic OA – Kellgren-Lawrence Grade III or IV",
            "Failed conservative management: minimum 6 months",
            "Significant functional impairment documented",
            "Failed at least 1 corticosteroid injection",
        ],
    },

    "post_hip_pt": {
        "procedure": "Post-Operative Physical Therapy – Hip",
        "cpt_codes": ["97110", "97112", "97140", "97530"],
        "avg_cost_network": 2100.00,
        "approval_criteria": [
            "Status post-hip arthroplasty (within 3 months)",
            "Physician prescription",
        ],
    },

    "cardiac_followup": {
        "procedure": "Post-CABG Cardiology Follow-Up",
        "cpt_codes": ["99214"],
        "avg_cost_network": 580.00,
        "approval_criteria": [
            "Status post-CABG (within 6 months of surgery)",
            "Medication reconciliation required",
            "Cardiac rehab enrollment documented",
        ],
    },

    "stress_echocardiogram": {
        "procedure": "Stress Echocardiogram (Dobutamine)",
        "cpt_codes": ["93350", "93351", "J1250"],
        "avg_cost_network": 2800.00,
        "approval_criteria": [
            "Clinical indication (chest pain, dyspnea, pre-op assessment)",
            "Resting EKG non-diagnostic",
            "Unable to perform exercise stress test",
        ],
    },

    "copd_exacerbation": {
        "procedure": "COPD Exacerbation – Inpatient Admission",
        "cpt_codes": ["99222", "99233", "94640"],
        "avg_cost_network": 15000.00,
        "approval_criteria": [
            "Documented acute respiratory failure (ABG or SpO2 criteria)",
            "Infectious trigger identified or presumptively treated",
            "GOLD Stage 2 or higher COPD diagnosis",
        ],
    },

    "echocardiogram": {
        "procedure": "Transthoracic Echocardiogram",
        "cpt_codes": ["93306"],
        "avg_cost_network": 1900.00,
        "approval_criteria": [
            "Clinical indication (HF, valve disease, pre-op, post-event)",
            "No recent echo within 6 months (unless clinical change documented)",
        ],
    },

    "medication_review": {
        "procedure": "Complex Medication Management Visit",
        "cpt_codes": ["99214", "99483"],
        "avg_cost_network": 380.00,
        "approval_criteria": [
            ">=5 active medications",
            ">=2 chronic conditions",
            "Medication reconciliation performed",
        ],
    },

    "shoulder_mri": {
        "procedure": "MRI Shoulder – Without Contrast",
        "cpt_codes": ["73221"],
        "avg_cost_network": 2200.00,
        "approval_criteria": [
            "Suspected rotator cuff tear, labral tear, or instability",
            "Failed 4 weeks conservative treatment OR acute traumatic injury",
            "X-ray non-diagnostic",
        ],
    },

    "rotator_cuff_repair": {
        "procedure": "Arthroscopic Rotator Cuff Repair",
        "cpt_codes": ["23427", "29826"],
        "avg_cost_network": 15200.00,
        "avg_cost_out_of_network": 21000.00,
        "conservative_treatment_required_months": 1,
        "required_imaging": ["MRI"],
        "approval_criteria": [
            "MRI-confirmed full-thickness rotator cuff tear",
            "Failed conservative management OR acute trauma with retraction >2 cm",
            "Functional impairment documented",
            "Age <70 (primary repair)",
        ],
    },

    "interim_pet_ct": {
        "procedure": "Interim PET/CT – Lymphoma Response Assessment",
        "cpt_codes": ["78815"],
        "avg_cost_network": 7200.00,
        "approval_criteria": [
            "Active lymphoma treatment ongoing",
            "NCCN-guideline-consistent interim assessment timing",
            "Oncologist attestation",
        ],
    },

    "post_spine_followup": {
        "procedure": "Post-Operative Spine Follow-Up with X-ray",
        "cpt_codes": ["99213", "72100"],
        "avg_cost_network": 520.00,
        "approval_criteria": [
            "Status post-spinal fusion (within 3 months)",
            "X-ray ordered for fusion assessment",
        ],
    },

    "cataract_surgery": {
        "procedure": "Phacoemulsification Cataract Surgery with IOL",
        "cpt_codes": ["66984"],
        "avg_cost_network": 4200.00,
        "approval_criteria": [
            "Documented visually significant cataract (BCVA <=20/40)",
            "Ophthalmologist attestation of medical necessity",
            "IOL power calculation documented",
        ],
    },

    "dermatology_excision": {
        "procedure": "Excisional Biopsy – Suspicious Skin Lesion",
        "cpt_codes": ["11603", "11604"],
        "avg_cost_network": 1450.00,
        "approval_criteria": [
            "Suspicious lesion with ABCDE criteria",
            "Dermatologist attestation",
            "Pathology review ordered",
        ],
    },

    "er_visit": {
        "procedure": "Emergency Room Visit",
        "cpt_codes": ["99284", "74177"],
        "avg_cost_network": 2600.00,
        "approval_criteria": [
            "Acute symptom onset within 48 hours",
            "Prudent layperson standard met",
        ],
    },

    "vaccination": {
        "procedure": "Routine Vaccination Administration",
        "cpt_codes": ["90471", "90715", "90686"],
        "avg_cost_network": 150.00,
        "approval_criteria": ["CDC-recommended immunization schedule followed", "Patient enrolled in plan"],
    },

    "prenatal_visit": {
        "procedure": "Routine Prenatal Visit",
        "cpt_codes": ["59425", "59426"],
        "avg_cost_network": 440.00,
        "approval_criteria": [
            "Confirmed pregnancy",
            "ACOG-recommended visit schedule followed",
            "OB/GYN attestation",
        ],
    },

    "delivery_maternity": {
        "procedure": "Vaginal Delivery with Epidural – Term",
        "cpt_codes": ["59400", "59410", "01967"],
        "avg_cost_network": 9800.00,
        "approval_criteria": [
            "Delivery at >=37 weeks gestation",
            "Admission by credentialed OB/GYN",
            "Hospital length of stay 1-2 days (vaginal) per medical necessity",
        ],
    },

    "colonoscopy": {
        "procedure": "Screening Colonoscopy",
        "cpt_codes": ["45378", "G0121"],
        "avg_cost_network": 2200.00,
        "approval_criteria": [
            "Age >=45 (average risk per USPSTF) OR high-risk criteria met",
            "No colonoscopy within 10 years (average risk)",
        ],
    },

    "sleep_study": {
        "procedure": "In-Lab Polysomnography with CPAP Titration",
        "cpt_codes": ["95810", "95811"],
        "avg_cost_network": 3800.00,
        "approval_criteria": [
            "High pre-test probability of OSA (ESS >=10, STOP-BANG >=3)",
            "Home sleep study failed or not indicated (comorbid CHF, COPD, central apnea)",
            "AASM-accredited sleep lab",
        ],
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# PROCEDURE CODE MAP (claim procedure -> guidelines key)
# ══════════════════════════════════════════════════════════════════════════════

PROCEDURE_GUIDELINE_MAP: Dict[str, str] = {
    "Knee Arthroscopy with Meniscectomy": "knee_arthroscopy",
    "Right Knee Arthroscopy": "knee_arthroscopy",
    "Post-Operative Complication Visit": "knee_arthroscopy",
    "Post-Operative Knee Physical Therapy (12 sessions)": "pt_knee_rehab",
    "Annual Wellness Examination": "annual_wellness",
    "R-CHOP Chemotherapy Cycle 3": "chemotherapy_rchop",
    "R-CHOP Chemotherapy Cycle 4": "chemotherapy_rchop",
    "Interim PET/CT Scan – Lymphoma Response Assessment": "interim_pet_ct",
    "Lumbar Spinal Fusion L4-S1": "lumbar_fusion",
    "Post-Operative Spine Follow-Up & X-ray": "post_spine_followup",
    "Emergency Appendectomy": "appendectomy_emergency",
    "Diabetes Management Comprehensive Visit": "diabetes_management",
    "Diabetic Foot Ulcer Debridement & Wound Care": "wound_care_diabetic",
    "Phacoemulsification Cataract Surgery – Left Eye": "cataract_surgery",
    "Excision of Suspicious Skin Lesion – Back": "dermatology_excision",
    "ER Visit – Abdominal Pain": "er_visit",
    "Post-CABG Cardiology Follow-Up": "cardiac_followup",
    "Stress Echocardiogram": "stress_echocardiogram",
    "Polysomnography – Sleep Study": "sleep_study",
    "Total Hip Replacement – Right": "hip_replacement",
    "Post-Operative Hip Physical Therapy (10 sessions)": "post_hip_pt",
    "Routine Vaccination – Tdap + Influenza": "vaccination",
    "Screening Colonoscopy": "colonoscopy",
    "COPD Exacerbation – Inpatient Admission (3 days)": "copd_exacerbation",
    "Transthoracic Echocardiogram": "echocardiogram",
    "Complex Medication Management Visit": "medication_review",
    "MRI Shoulder – Right (without contrast)": "shoulder_mri",
    "Arthroscopic Rotator Cuff Repair – Right Shoulder": "rotator_cuff_repair",
    "Prenatal Visit – 28-week Gestation": "prenatal_visit",
    "Vaginal Delivery with Epidural – Term": "delivery_maternity",
}