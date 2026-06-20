import os
from PIL import Image
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


def create_splits(data_dir, num_classes, class_names=None, has_predefined_splits=False, seed=42):
    if has_predefined_splits:
        return _create_predefined_splits(data_dir, class_names, seed)
    return _create_random_splits(data_dir, num_classes, class_names, seed)


def _create_random_splits(data_dir, num_classes, class_names, seed):
    samples, class_to_idx = _discover_samples(data_dir, class_names)

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


def _create_predefined_splits(data_dir, class_names, seed):
    num_classes = len(class_names)

    val_dir = _find_split_dir(data_dir, ["val", "valid", "validation"])
    train_dir = _find_split_dir(data_dir, ["train"])
    test_dir = _find_split_dir(data_dir, ["test"])

    if val_dir is not None and train_dir is not None and test_dir is not None:
        return _create_full_predefined_splits(train_dir, val_dir, test_dir, class_names, num_classes)

    test_a_dir = _find_split_dir(data_dir, ["test-a"])
    if train_dir is not None and test_a_dir is not None:
        train_all, _ = _discover_samples(train_dir, class_names)
        test_samples, _ = _discover_samples(test_a_dir, class_names)

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


def _create_full_predefined_splits(train_dir, val_dir, test_dir, class_names, num_classes):
    train_samples, _ = _discover_samples(train_dir, class_names)
    val_samples, _ = _discover_samples(val_dir, class_names)
    test_samples, _ = _discover_samples(test_dir, class_names)

    _print_split_counts("Train", train_samples, num_classes)
    _print_split_counts("Val", val_samples, num_classes)
    _print_split_counts("Test", test_samples, num_classes)

    return train_samples, val_samples, test_samples


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
