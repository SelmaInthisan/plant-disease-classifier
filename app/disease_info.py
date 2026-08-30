"""Information shown by the binary Healthy/Diseased classifier."""

DISEASE_DATABASE = {
    "Healthy": {
        "prediction": "Healthy",
        "crop": "Foliage Crop",
        "condition": "Healthy Plant Leaf",
        "display_name": "Healthy Leaf Condition",
        "is_healthy": True,
        "pathogen_type": "None",
        "severity": "None",
        "symptoms": [
            "Uniform green leaf pigmentation with no obvious disease lesions.",
            "No prominent necrotic spots, chlorotic rings, or visible fungal growth.",
        ],
        "causes": "Healthy appearance is consistent with normal plant growth and adequate growing conditions.",
        "prevention": [
            "Maintain appropriate irrigation and avoid prolonged leaf wetness.",
            "Provide adequate sunlight and air circulation.",
            "Monitor leaves regularly for changes or new symptoms.",
        ],
        "organic_treatment": "No corrective treatment is indicated from a Healthy prediction. Continue routine crop care.",
        "chemical_treatment": "No chemical treatment is indicated from a Healthy prediction.",
    },
    "Diseased": {
        "prediction": "Diseased",
        "crop": "Foliage Crop",
        "condition": "Diseased Plant Leaf",
        "display_name": "Diseased Leaf Condition",
        "is_healthy": False,
        "pathogen_type": "Potential pathogenic microorganism (fungal / bacterial / oomycete)",
        "severity": "Not determined by this binary model",
        "symptoms": [
            "Visible discoloration, lesions, spotting, necrotic tissue, or other abnormal leaf patterns may be present.",
            "The exact disease cannot be identified by this binary classifier.",
        ],
        "causes": "Possible contributors include elevated humidity, prolonged leaf wetness, poor air circulation, or contaminated soil splash; the exact cause is not determined by this model.",
        "prevention": [
            "Avoid prolonged leaf wetness and improve air circulation.",
            "Use root-zone or drip irrigation where practical.",
            "Remove severely affected plant material and sanitize tools.",
            "Monitor plants regularly and seek disease-specific diagnosis when needed.",
        ],
        "organic_treatment": "Use disease-management measures appropriate to the confirmed crop and disease. Do not treat this binary prediction as a confirmed pathogen diagnosis.",
        "chemical_treatment": "Any fungicide or bactericide should be selected only after confirming the crop, disease, local label requirements, and safety interval.",
    },
}

def get_disease_info(class_name: str) -> dict:
    if class_name in DISEASE_DATABASE:
        return DISEASE_DATABASE[class_name]
    return DISEASE_DATABASE["Healthy" if "healthy" in class_name.lower() else "Diseased"]
