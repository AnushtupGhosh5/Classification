import os
import csv
import random
from PIL import Image
from torch.utils.data import Dataset

CLASS_NAMES = ["Benign", "Malignant"]
NUM_CLASSES = 2


def load_labels_from_csv(csv_path):
    patient_labels = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            patient_labels[row["patient_name"].strip()] = int(row["histo_label"].strip())
    return patient_labels


def load_all_images(data_dir):
    samples = []
    for fname in sorted(os.listdir(data_dir)):
        if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
            patient_id = fname.split("_")[0]
            samples.append((os.path.join(data_dir, fname), patient_id))
    return samples


def stratified_patient_split(samples, train_ratio=0.7, val_ratio=0.15, seed=42):
    patient_to_images = {}
    patient_to_label = {}
    for img_path, patient_id in samples:
        if patient_id not in patient_to_images:
            patient_to_images[patient_id] = []
        patient_to_images[patient_id].append(img_path)
        if patient_id not in patient_to_label:
            patient_to_label[patient_id] = None

    for img_path, patient_id in samples:
        pass

    return patient_to_images, patient_to_label


def create_splits(data_dir, train_ratio=0.7, val_ratio=0.15, seed=42):
    batches = []
    for batch_dir_name in sorted(os.listdir(data_dir)):
        batch_path = os.path.join(data_dir, batch_dir_name)
        if not os.path.isdir(batch_path):
            continue
        for sub in sorted(os.listdir(batch_path)):
            sub_path = os.path.join(batch_path, sub)
            if not os.path.isdir(sub_path):
                continue
            label_csv = None
            dataset_dir = None
            for item in os.listdir(sub_path):
                item_path = os.path.join(sub_path, item)
                if item.endswith("_label.csv"):
                    label_csv = item_path
                elif os.path.isdir(item_path) and item == "dataset":
                    dataset_dir = item_path
            if label_csv and dataset_dir:
                batches.append((label_csv, dataset_dir))

    patient_labels = {}
    all_samples = []

    for batch_idx, (label_csv, dataset_dir) in enumerate(batches):
        batch_prefix = f"b{batch_idx}"
        batch_labels = load_labels_from_csv(label_csv)
        namespaced_labels = {f"{batch_prefix}_{pid}": label for pid, label in batch_labels.items()}
        patient_labels.update(namespaced_labels)
        images = load_all_images(dataset_dir)
        namespaced_images = [(path, f"{batch_prefix}_{pid}") for path, pid in images]
        all_samples.extend(namespaced_images)

    valid_samples = []
    for img_path, patient_id in all_samples:
        if patient_id in patient_labels:
            valid_samples.append((img_path, patient_id, patient_labels[patient_id]))

    patient_to_images = {}
    patient_to_label = {}
    for img_path, patient_id, label in valid_samples:
        if patient_id not in patient_to_images:
            patient_to_images[patient_id] = []
            patient_to_label[patient_id] = label
        patient_to_images[patient_id].append((img_path, label))

    patients_by_class = {0: [], 1: []}
    for pid, label in patient_to_label.items():
        patients_by_class[label].append(pid)

    rng = random.Random(seed)
    rng.shuffle(patients_by_class[0])
    rng.shuffle(patients_by_class[1])

    train_patients = set()
    val_patients = set()
    test_patients = set()

    for cls in [0, 1]:
        pids = patients_by_class[cls]
        n = len(pids)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_patients.update(pids[:n_train])
        val_patients.update(pids[n_train:n_train + n_val])
        test_patients.update(pids[n_train + n_val:])

    train_samples = []
    val_samples = []
    test_samples = []

    for pid in train_patients:
        for img_path, label in patient_to_images[pid]:
            train_samples.append((img_path, label))
    for pid in val_patients:
        for img_path, label in patient_to_images[pid]:
            val_samples.append((img_path, label))
    for pid in test_patients:
        for img_path, label in patient_to_images[pid]:
            test_samples.append((img_path, label))

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    rng.shuffle(test_samples)

    print(f"Patients - Train: {len(train_patients)}, Val: {len(val_patients)}, Test: {len(test_patients)}")
    print(f"Images   - Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")

    for split_name, samples in [("Train", train_samples), ("Val", val_samples), ("Test", test_samples)]:
        counts = {0: 0, 1: 0}
        for _, label in samples:
            counts[label] += 1
        print(f"  {split_name} - Benign: {counts[0]}, Malignant: {counts[1]}")

    return train_samples, val_samples, test_samples


class ThyroidDataset(Dataset):
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
        counts = {0: 0, 1: 0}
        for _, label in self.samples:
            counts[label] += 1
        return counts
