import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DATA_ROOT = os.path.join(_PROJECT_ROOT, "data")

DATASET_REGISTRY = {
    "all4": {
        "data_dir": os.path.join(
            _DATA_ROOT,
            "ALL(Acute Lymphoblastic Leukemia) 4 classification",
            "Acute Lymphoblastic Leukemia (ALL)  classification",
            "Acute Lymphoblastic Leukemia (ALL)  classification",
            "Original",
        ),
        "class_names": ["Benign", "Early", "Pre", "Pro"],
        "num_classes": 4,
        "has_predefined_splits": False,
    },
    "lymphoma": {
        "data_dir": os.path.join(
            _DATA_ROOT,
            "Malignant Lymphoma Classification",
        ),
        "class_names": ["CLL", "FL", "MCL"],
        "num_classes": 3,
        "has_predefined_splits": False,
    },
    "pbc8": {
        "data_dir": os.path.join(
            _DATA_ROOT,
            "PBC",
            "PBC_dataset_normal_DIB",
        ),
        "class_names": [
            "basophil", "eosinophil", "erythroblast", "ig",
            "lymphocyte", "monocyte", "neutrophil", "platelet",
        ],
        "num_classes": 8,
        "has_predefined_splits": False,
    },
    "raabin": {
        "data_dir": os.path.join(
            _DATA_ROOT,
            "Raabin-WBC Dataset Final",
        ),
        "class_names": ["Basophil", "Eosinophil", "Lymphocyte", "Monocyte", "Neutrophil"],
        "num_classes": 5,
        "has_predefined_splits": True,
    },
}


def get_dataset_config(name):
    if name not in DATASET_REGISTRY:
        available = ", ".join(DATASET_REGISTRY.keys())
        raise ValueError(f"Unknown dataset '{name}'. Available: {available}")
    return DATASET_REGISTRY[name]
