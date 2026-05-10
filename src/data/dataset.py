import os
import xml.etree.ElementTree as ET
from PIL import Image
from torch.utils.data import Dataset

CLASS_NAMES = ["Benign", "Malignant"]
NUM_CLASSES = 2


def parse_voc_annotation(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    obj = root.find("object")
    label = int(obj.find("name").text)
    bbox = obj.find("bndbox")
    xmin = int(bbox.find("xmin").text)
    ymin = int(bbox.find("ymin").text)
    xmax = int(bbox.find("xmax").text)
    ymax = int(bbox.find("ymax").text)
    return label, (xmin, ymin, xmax, ymax)


def crop_with_padding(image, bbox, padding_ratio=0.1):
    w, h = image.size
    xmin, ymin, xmax, ymax = bbox
    bw = xmax - xmin
    bh = ymax - ymin
    pad_w = int(bw * padding_ratio)
    pad_h = int(bh * padding_ratio)
    xmin = max(0, xmin - pad_w)
    ymin = max(0, ymin - pad_h)
    xmax = min(w, xmax + pad_w)
    ymax = min(h, ymax + pad_h)
    return image.crop((xmin, ymin, xmax, ymax))


def create_splits(data_dir, bbox_padding=0.1, seed=42):
    jpeg_dir = os.path.join(data_dir, "JPEGImages")
    anno_dir = os.path.join(data_dir, "Annotations")
    splits_dir = os.path.join(data_dir, "ImageSets", "Main")

    splits = {}
    for split_name in ["train", "val", "test"]:
        split_file = os.path.join(splits_dir, f"{split_name}.txt")
        samples = []
        with open(split_file, "r") as f:
            for line in f:
                img_id = line.strip()
                if not img_id:
                    continue
                img_path = os.path.join(jpeg_dir, f"{img_id}.jpg")
                xml_path = os.path.join(anno_dir, f"{img_id}.xml")
                if not os.path.exists(img_path) or not os.path.exists(xml_path):
                    continue
                label, bbox = parse_voc_annotation(xml_path)
                samples.append((img_path, label, bbox))
        splits[split_name] = samples

    train_samples = splits["train"]
    val_samples = splits["val"]
    test_samples = splits["test"]

    for split_name, samples in [("Train", train_samples), ("Val", val_samples), ("Test", test_samples)]:
        counts = {0: 0, 1: 0}
        for _, label, _ in samples:
            counts[label] += 1
        print(f"  {split_name}: {len(samples)} images | Benign: {counts[0]}, Malignant: {counts[1]}")

    return train_samples, val_samples, test_samples


class ThyroidDataset(Dataset):
    def __init__(self, samples, transform=None, bbox_padding=0.1):
        self.samples = samples
        self.transform = transform
        self.bbox_padding = bbox_padding

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, bbox = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = crop_with_padding(image, bbox, self.bbox_padding)
        if self.transform:
            image = self.transform(image)
        return image, label

    def get_class_counts(self):
        counts = {0: 0, 1: 0}
        for _, label, _ in self.samples:
            counts[label] += 1
        return counts
