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
    "milk10k": {
        "data_dir": os.path.join(
            _DATA_ROOT,
            "MILK10k",
            "MILK10K-split",
            "kaggle",
            "working",
            "MILK10K-split",
        ),
        "class_names": [
            "AKIEC", "BCC", "BEN_OTH", "BKL", "DF", "INF",
            "MAL_OTH", "MEL", "NV", "SCCKA", "VASC",
        ],
        "num_classes": 11,
        "has_predefined_splits": True,
    },
    "isic17": {
        "data_dir": os.path.join(_DATA_ROOT, "ISIC_17"),
        "class_names": ["melanoma", "nevus", "seborrheic_keratosis"],
        "num_classes": 3,
        "has_predefined_splits": True,
        "original_class_counts": [374, 1372, 254],  # melanoma, nevus, seborrheic_keratosis
        # ISIC17 has a very small, noisy validation set (150 imgs across 3
        # classes). Late in training the model becomes overconfident: correct
        # predictions -> prob ~1 and the inherently-ambiguous samples it can
        # never fit -> prob ~0, whose exponentially growing loss dominates the
        # mean. Reported val loss then rises even as accuracy improves.
        #
        # Bi-Tempered Logistic Loss directly addresses this by capping the
        # maximum per-sample loss (bounded log at t2=0.4), so mislabelled or
        # genuinely ambiguous samples cannot dominate the average.  EMA smooths
        # the reported val curve, early stopping halts near the minimum, and
        # grad clipping stabilises the late high-LR cosine schedule.
        "training_overrides": {
            "loss": "bi_tempered",
            "label_smoothing": 0.15,
            "ema": True,
            "ema_decay": 0.999,
            "early_stopping": True,
            "es_patience": 15,
            "es_min_delta": 0.0,
            "grad_clip": 1.0,
        },
    },
}


def get_dataset_config(name):
    if name not in DATASET_REGISTRY:
        available = ", ".join(DATASET_REGISTRY.keys())
        raise ValueError(f"Unknown dataset '{name}'. Available: {available}")
    config = DATASET_REGISTRY[name].copy()
    if "original_class_counts" in config:
        config["original_class_counts"] = config["original_class_counts"]
    return config
