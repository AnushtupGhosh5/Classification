import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DATA_ROOT = os.environ.get("CLASSIFICATION_DATA_ROOT", os.path.join(_PROJECT_ROOT, "data"))

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
        "training_overrides": {
            "early_stopping": True,
            "es_patience": 15,
            "es_min_delta": 0.0,
        },
    },
    "isic16": {
        "data_dir": os.path.join(_DATA_ROOT, "ISIC16"),
        "class_names": ["benign", "malignant"],
        "num_classes": 2,
        "has_predefined_splits": True,
    },
    "isic17": {
        "data_dir": os.path.join(_DATA_ROOT, "ISIC_17"),
        "class_names": ["melanoma", "nevus", "seborrheic_keratosis"],
        "num_classes": 3,
        "has_predefined_splits": True,
        "original_class_counts": [374, 1372, 254],
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
            # Original ISIC17 training distribution before offline
            # augmentation: melanoma=374, nevus=1372, seborrheic_keratosis=254.
            # Keep original counts available for experiments, but disable
            # weighting by default. The train folders are already heavily
            # augmented and nearly balanced, and even sqrt weighting was
            # forcing minority classes hard enough to hurt test accuracy.
            "class_weight_counts": [374, 1372, 254],
            "class_weight_power": 0.0,
            "label_smoothing": 0.15,
            "ema": True,
            "ema_decay": 0.999,
            "early_stopping": True,
            "es_patience": 15,
            "es_min_delta": 0.0,
            "grad_clip": 1.0,
        },
    },
    "isic18": {
        "data_dir": os.path.join(_DATA_ROOT, "isic18"),
        "class_names": ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"],
        "num_classes": 7,
        "has_predefined_splits": True,
        "split_dirs": {
            "train": os.path.join("ham10000_80_20", "ham10000_80_20", "train_dir"),
            "val": os.path.join("Valid", "kaggle", "working", "Valid"),
            "test": os.path.join("Test", "kaggle", "working", "Test"),
        },
        "validate_images": True,
        "training_overrides": {
            "early_stopping": True,
            "es_patience": 15,
            "es_min_delta": 0.0,
        },
    },
    "isic19": {
        "data_dir": os.path.join(_DATA_ROOT, "isic19"),
        "class_names": ["AK", "BCC", "BKL", "DF", "MEL", "NV", "SCC", "VASC"],
        "num_classes": 8,
        "has_predefined_splits": True,
        "split_dirs": {
            "train": os.path.join(
                "Train_Validation_dataset",
                "content",
                "Train_Validation_dataset",
                "train",
            ),
            "val": os.path.join(
                "Train_Validation_dataset",
                "content",
                "Train_Validation_dataset",
                "val",
            ),
            "test": os.path.join(
                "Test_dataset",
                "content",
                "test_dataset2",
            ),
        },
        "train_sample_limit": 5000,
        "train_sampling_strategy": "balanced_random",
        "validate_images": True,
        "fallback_val_from_train": True,
        "training_overrides": {
            "early_stopping": True,
            "es_patience": 15,
            "es_min_delta": 0.0,
        },
    },
}


def get_dataset_config(name):
    if name not in DATASET_REGISTRY:
        available = ", ".join(DATASET_REGISTRY.keys())
        raise ValueError(f"Unknown dataset '{name}'. Available: {available}")
    return DATASET_REGISTRY[name]
