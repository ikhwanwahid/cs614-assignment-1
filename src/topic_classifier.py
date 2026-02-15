"""Keyword-based heuristic medical topic tagger.

Assigns each MedQA question to one of ~13 medical topics based on keyword
matching. Used for per-topic accuracy breakdown in evaluation.

Limitation: This is a heuristic -- questions may be misclassified when they
span multiple topics or use uncommon terminology.
"""

TOPIC_KEYWORDS: dict[str, list[str]] = {
    "Cardiology": [
        "heart", "cardiac", "ecg", "ekg", "arrhythmia", "hypertension",
        "murmur", "myocardial", "coronary", "atrial", "ventricular",
        "aortic", "mitral", "pericardi", "endocardi", "chest pain",
        "heart failure", "angina", "stemi", "nstemi",
    ],
    "Pulmonology": [
        "lung", "pulmonary", "respiratory", "pneumonia", "asthma",
        "copd", "bronch", "pleural", "dyspnea", "cough", "emphysema",
        "tuberculosis", "sarcoid", "pulmonary embolism",
    ],
    "Gastroenterology": [
        "liver", "hepat", "gastric", "intestin", "colon", "pancrea",
        "biliary", "gallbladder", "esophag", "bowel", "cirrhosis",
        "jaundice", "diarrhea", "constipat", "crohn", "colitis",
        "celiac", "abdominal pain",
    ],
    "Neurology": [
        "brain", "neuro", "seizure", "stroke", "cerebr", "meningit",
        "headache", "cranial nerve", "multiple sclerosis", "parkinson",
        "alzheimer", "neuropath", "spinal cord", "dementia",
        "encephalit", "epilepsy",
    ],
    "Nephrology": [
        "kidney", "renal", "creatinine", "glomerul", "dialysis",
        "nephr", "proteinuria", "hematuria", "uremia", "bun",
        "electrolyte", "hyperkalemia", "hyponatremia",
    ],
    "Infectious Disease": [
        "infection", "bacteria", "viral", "fungal", "antibiotic",
        "hiv", "aids", "sepsis", "fever", "abscess", "mrsa",
        "streptococ", "staphylococ", "malaria", "hepatitis",
    ],
    "Endocrinology": [
        "thyroid", "diabetes", "insulin", "adrenal", "pituitary",
        "hormone", "cortisol", "cushing", "addison", "hyperglycemia",
        "hypoglycemia", "a1c", "testosterone", "estrogen",
    ],
    "Obstetrics/Gynecology": [
        "pregnant", "gestation", "uterus", "ovary", "cervical",
        "menstrual", "fetus", "preeclampsia", "eclampsia", "labor",
        "delivery", "contracepti", "amenorrhea", "postpartum",
        "trimester", "placenta",
    ],
    "Oncology": [
        "cancer", "tumor", "malignant", "metast", "carcinoma",
        "lymphoma", "leukemia", "chemotherapy", "radiation therapy",
        "biopsy", "mass", "neoplasm", "oncolog",
    ],
    "Pharmacology": [
        "mechanism of action", "adverse effect", "side effect",
        "contraindic", "drug interaction", "pharmacokinetic",
        "pharmacodynamic", "dosage", "toxicity", "overdose",
        "antidote", "receptor",
    ],
    "Psychiatry": [
        "depression", "anxiety", "psychosis", "schizophren",
        "bipolar", "ssri", "psychiatric", "suicid", "hallucination",
        "delusion", "ptsd", "ocd", "panic", "mood",
    ],
    "Hematology": [
        "anemia", "coagulation", "platelet", "hemoglobin",
        "bleeding", "thrombocyt", "leukocyt", "lymphocyt",
        "sickle cell", "thalassemia", "hemophilia", "dvt",
        "anticoagul",
    ],
}


def classify_topic(question_text: str) -> str:
    """Classify a question into a medical topic based on keyword matching.

    Returns the topic with the most keyword hits.
    Falls back to 'Other' if no keywords match.
    """
    text_lower = question_text.lower()
    scores: dict[str, int] = {}

    for topic, keywords in TOPIC_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[topic] = score

    if not scores:
        return "Other"
    return max(scores, key=scores.get)


def classify_dataset(dataset) -> list[str]:
    """Classify all questions in a dataset. Returns list of topic strings."""
    return [classify_topic(example["sent1"]) for example in dataset]
