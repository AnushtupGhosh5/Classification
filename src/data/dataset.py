import os
import random
import csv
from collections import defaultdict
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _discover_samples(root_dir, class_names=None):
    if class_names is None:
        class_names = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])

    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    samples = []

    for class_name in class_names:
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        label = class_to_idx[class_name]
        for fname in sorted(os.listdir(class_dir)):
            ext = os.path.splitext(fname)[1].lower()
            if ext in VALID_EXTENSIONS:
                samples.append((os.path.join(class_dir, fname), label))

    return samples, class_to_idx


def create_splits(
    data_dir,
    num_classes,
    class_names=None,
    has_predefined_splits=False,
    seed=42,
    split_dirs=None,
    train_sample_limit=None,
    train_sampling_strategy="balanced_random",
    validate_images=False,
    fallback_val_from_train=False,
):
    if has_predefined_splits:
        return _create_predefined_splits(
            data_dir,
            class_names,
            seed,
            split_dirs=split_dirs,
            train_sample_limit=train_sample_limit,
            train_sampling_strategy=train_sampling_strategy,
            validate_images=validate_images,
            fallback_val_from_train=fallback_val_from_train,
        )
    return _create_random_splits(data_dir, num_classes, class_names, seed, validate_images)


def _create_random_splits(data_dir, num_classes, class_names, seed, validate_images=False):
    samples, class_to_idx = _discover_samples(data_dir, class_names)
    samples = _filter_invalid_images(samples, "All", validate_images)

    labels = [s[1] for s in samples]
    train_val, test_samples = train_test_split(
        samples, test_size=0.15, random_state=seed, stratify=labels,
    )
    train_val_labels = [s[1] for s in train_val]
    train_samples, val_samples = train_test_split(
        train_val, test_size=0.176, random_state=seed, stratify=train_val_labels,
    )

    _print_split_counts("Train", train_samples, num_classes)
    _print_split_counts("Val", val_samples, num_classes)
    _print_split_counts("Test", test_samples, num_classes)

    return train_samples, val_samples, test_samples


def _find_split_dir(data_dir, candidates):
    actual = {d.lower(): d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))}
    for candidate in candidates:
        if candidate.lower() in actual:
            return os.path.join(data_dir, actual[candidate.lower()])
    return None


def _resolve_split_dir(data_dir, split_path):
    if split_path is None:
        return None
    if os.path.isabs(split_path):
        return split_path
    return os.path.join(data_dir, split_path)


def _create_predefined_splits(
    data_dir,
    class_names,
    seed,
    split_dirs=None,
    train_sample_limit=None,
    train_sampling_strategy="balanced_random",
    validate_images=False,
    fallback_val_from_train=False,
):
    num_classes = len(class_names)

    if split_dirs:
        train_dir = _resolve_split_dir(data_dir, split_dirs.get("train"))
        val_dir = _resolve_split_dir(data_dir, split_dirs.get("val"))
        test_dir = _resolve_split_dir(data_dir, split_dirs.get("test"))
    else:
        val_dir = _find_split_dir(data_dir, ["val", "valid", "validation"])
        train_dir = _find_split_dir(data_dir, ["train"])
        test_dir = _find_split_dir(data_dir, ["test"])

    if val_dir is not None and train_dir is not None and test_dir is not None:
        for split_name, split_dir in [("train", train_dir), ("val", val_dir), ("test", test_dir)]:
            if not os.path.isdir(split_dir):
                raise FileNotFoundError(f"Configured {split_name} split directory does not exist: {split_dir}")
        return _create_full_predefined_splits(
            train_dir,
            val_dir,
            test_dir,
            class_names,
            num_classes,
            seed=seed,
            train_sample_limit=train_sample_limit,
            train_sampling_strategy=train_sampling_strategy,
            validate_images=validate_images,
            fallback_val_from_train=fallback_val_from_train,
        )

    test_a_dir = _find_split_dir(data_dir, ["test-a"])
    if train_dir is not None and test_a_dir is not None:
        train_all, _ = _discover_samples(train_dir, class_names)
        test_samples, _ = _discover_samples(test_a_dir, class_names)
        train_all = _filter_invalid_images(train_all, "Train", validate_images)
        test_samples = _filter_invalid_images(test_samples, "Test", validate_images)

        train_labels = [s[1] for s in train_all]
        train_samples, val_samples = train_test_split(
            train_all, test_size=0.2, random_state=seed, stratify=train_labels,
        )

        _print_split_counts("Train", train_samples, num_classes)
        _print_split_counts("Val", val_samples, num_classes)
        _print_split_counts("Test", test_samples, num_classes)

        return train_samples, val_samples, test_samples

    raise FileNotFoundError(
        f"Could not find predefined split folders in {data_dir}. "
        f"Expected train/val(valid)/test or train/test-a directories."
    )


def _create_full_predefined_splits(
    train_dir,
    val_dir,
    test_dir,
    class_names,
    num_classes,
    seed=42,
    train_sample_limit=None,
    train_sampling_strategy="balanced_random",
    validate_images=False,
    fallback_val_from_train=False,
):
    train_samples, _ = _discover_samples(train_dir, class_names)
    val_samples, _ = _discover_samples(val_dir, class_names)
    test_samples, _ = _discover_samples(test_dir, class_names)

    train_samples = _filter_invalid_images(train_samples, "Train", validate_images)
    val_samples = _filter_invalid_images(val_samples, "Val", validate_images)
    test_samples = _filter_invalid_images(test_samples, "Test", validate_images)

    if fallback_val_from_train and _has_missing_classes(val_samples, num_classes):
        print("  Val: missing one or more classes after validation; rebuilding val split from train")
        train_labels = [s[1] for s in train_samples]
        train_samples, val_samples = train_test_split(
            train_samples, test_size=0.2, random_state=seed, stratify=train_labels,
        )

    if train_sample_limit is not None and len(train_samples) > train_sample_limit:
        train_samples = _sample_train_subset(
            train_samples,
            num_classes,
            train_sample_limit,
            seed,
            train_sampling_strategy,
        )

    _print_split_counts("Train", train_samples, num_classes)
    _print_split_counts("Val", val_samples, num_classes)
    _print_split_counts("Test", test_samples, num_classes)

    return train_samples, val_samples, test_samples


def _is_readable_image(path):
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (UnidentifiedImageError, OSError, ValueError):
        return False


def _filter_invalid_images(samples, split_name, validate_images=False):
    if not validate_images:
        return samples

    valid = []
    invalid = []
    for path, label in samples:
        if _is_readable_image(path):
            valid.append((path, label))
        else:
            invalid.append(path)

    if invalid:
        print(f"  {split_name}: skipped {len(invalid)} unreadable image files")
        preview = ", ".join(os.path.basename(path) for path in invalid[:5])
        suffix = " ..." if len(invalid) > 5 else ""
        print(f"    Invalid examples: {preview}{suffix}")

    return valid


def _has_missing_classes(samples, num_classes):
    labels = {label for _, label in samples}
    return any(label not in labels for label in range(num_classes))


def _sample_train_subset(samples, num_classes, limit, seed, strategy):
    if strategy == "random":
        rng = random.Random(seed)
        selected = rng.sample(samples, limit)
        return sorted(selected, key=lambda item: item[0])
    if strategy != "balanced_random":
        raise ValueError(
            f"Unknown train sampling strategy '{strategy}'. "
            "Available: balanced_random, random"
        )
    return _sample_balanced_random_subset(samples, num_classes, limit, seed)


def _sample_balanced_random_subset(samples, num_classes, limit, seed):
    by_class = defaultdict(list)
    for path, label in samples:
        by_class[label].append((path, label))

    rng = random.Random(seed)
    for class_samples in by_class.values():
        rng.shuffle(class_samples)

    original_counts = {label: len(by_class.get(label, [])) for label in range(num_classes)}
    target_counts = _balanced_sample_counts(original_counts, limit, num_classes)

    selected = []
    for label in range(num_classes):
        selected.extend(by_class.get(label, [])[:target_counts.get(label, 0)])

    rng.shuffle(selected)
    print(
        f"  Train subset: selected {len(selected)} of {len(samples)} images "
        f"(limit={limit}, strategy=balanced_random, seed={seed})"
    )
    return selected


def _balanced_sample_counts(class_counts, limit, num_classes):
    target_total = min(limit, sum(class_counts.values()))
    targets = {label: 0 for label in range(num_classes)}
    remaining_labels = {label for label in range(num_classes) if class_counts.get(label, 0) > 0}
    remaining_total = target_total

    while remaining_labels and remaining_total > 0:
        base = remaining_total // len(remaining_labels)
        extra = remaining_total % len(remaining_labels)
        if base == 0:
            for label in sorted(remaining_labels)[:extra]:
                targets[label] += 1
            break

        progressed = False
        for offset, label in enumerate(sorted(remaining_labels)):
            desired = base + (1 if offset < extra else 0)
            capacity = class_counts[label] - targets[label]
            take = min(desired, capacity)
            targets[label] += take
            remaining_total -= take
            progressed = progressed or take > 0

        remaining_labels = {
            label for label in remaining_labels
            if targets[label] < class_counts[label]
        }
        if not progressed:
            break

    return targets


def _print_split_counts(split_name, samples, num_classes):
    counts = {i: 0 for i in range(num_classes)}
    for _, label in samples:
        counts[label] = counts.get(label, 0) + 1
    parts = [f"class{i}: {counts.get(i, 0)}" for i in range(num_classes)]
    print(f"  {split_name}: {len(samples)} images | {', '.join(parts)}")


class FolderDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

    def get_class_counts(self):
        counts = {}
        for _, label in self.samples:
            counts[label] = counts.get(label, 0) + 1
        return counts


def create_milk10k_lesion_splits(
    images_dir,
    ground_truth_csv,
    metadata_csv,
    class_names,
    seed=42,
    val_fraction=0.10,
    test_fraction=0.20,
):
    """Create deterministic, stratified MILK10k splits at lesion level.

    Each returned sample contains the clinical and dermoscopic image belonging
    to the same lesion.  Splitting *before* expanding image pairs prevents the
    cross-split lesion leakage present in the old third-party folder split.
    The returned test partition is a local development holdout from the public
    training set; it is not the hidden MILK10k Benchmark test set.
    """
    if val_fraction <= 0 or test_fraction <= 0:
        raise ValueError("MILK10k validation and local-test fractions must be positive")
    if val_fraction + test_fraction >= 1:
        raise ValueError("MILK10k validation + local-test fractions must be below 1")

    class_to_idx = {name: index for index, name in enumerate(class_names)}
    with open(ground_truth_csv, newline="") as file:
        truth_rows = list(csv.DictReader(file))
    with open(metadata_csv, newline="") as file:
        metadata_rows = list(csv.DictReader(file))

    labels_by_lesion = {}
    for row in truth_rows:
        positive = [name for name in class_names if float(row.get(name, 0.0)) >= 0.5]
        if len(positive) != 1:
            raise ValueError(
                f"Expected one diagnosis for {row.get('lesion_id')}, got {positive}"
            )
        labels_by_lesion[row["lesion_id"]] = class_to_idx[positive[0]]

    images_by_lesion = defaultdict(dict)
    for row in metadata_rows:
        lesion_id = row["lesion_id"]
        image_type = row["image_type"].strip().lower()
        if image_type == "clinical: close-up":
            modality = "clinical"
        elif image_type == "dermoscopic":
            modality = "dermoscopic"
        else:
            continue
        images_by_lesion[lesion_id][modality] = os.path.join(
            images_dir, lesion_id, f"{row['isic_id']}.jpg",
        )

    samples = []
    missing = []
    for lesion_id, label in sorted(labels_by_lesion.items()):
        pair = images_by_lesion.get(lesion_id, {})
        clinical = pair.get("clinical")
        dermoscopic = pair.get("dermoscopic")
        if not clinical or not dermoscopic or not os.path.isfile(clinical) or not os.path.isfile(dermoscopic):
            missing.append(lesion_id)
            continue
        samples.append((clinical, dermoscopic, label, lesion_id))
    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(
            f"MILK10k has {len(missing)} incomplete lesion pairs; examples: {preview}"
        )

    labels = [sample[2] for sample in samples]
    train_val, test_samples = train_test_split(
        samples,
        test_size=test_fraction,
        random_state=seed,
        stratify=labels,
    )
    relative_val_fraction = val_fraction / (1.0 - test_fraction)
    train_val_labels = [sample[2] for sample in train_val]
    train_samples, val_samples = train_test_split(
        train_val,
        test_size=relative_val_fraction,
        random_state=seed,
        stratify=train_val_labels,
    )

    _print_paired_split_counts("Train lesions", train_samples, len(class_names))
    _print_paired_split_counts("Val lesions", val_samples, len(class_names))
    _print_paired_split_counts("Local test lesions", test_samples, len(class_names))
    train_ids = {sample[3] for sample in train_samples}
    val_ids = {sample[3] for sample in val_samples}
    test_ids = {sample[3] for sample in test_samples}
    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise RuntimeError("MILK10k lesion-grouped split unexpectedly overlaps")
    print("  MILK10k grouping audit: 0 lesion IDs cross splits")
    return train_samples, val_samples, test_samples


def _print_paired_split_counts(split_name, samples, num_classes):
    counts = {index: 0 for index in range(num_classes)}
    for _, _, label, _ in samples:
        counts[label] += 1
    parts = [f"class{index}: {counts[index]}" for index in range(num_classes)]
    print(f"  {split_name}: {len(samples)} paired lesions | {', '.join(parts)}")


class PairedLesionDataset(Dataset):
    """Return a clinical/dermoscopic pair as ``[2, C, H, W]``."""

    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        clinical_path, dermoscopic_path, label, _lesion_id = self.samples[index]
        clinical = Image.open(clinical_path).convert("RGB")
        dermoscopic = Image.open(dermoscopic_path).convert("RGB")
        if self.transform:
            clinical = self.transform(clinical)
            dermoscopic = self.transform(dermoscopic)
        import torch
        return torch.stack((clinical, dermoscopic), dim=0), label

    def get_class_counts(self):
        counts = {}
        for _, _, label, _ in self.samples:
            counts[label] = counts.get(label, 0) + 1
        return counts
