import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.utils import evaluate_model


def evaluate_all_splits(model, train_loader, val_loader, test_loader, criterion, device, num_classes):
    results = {}

    for split_name, loader in [("train", train_loader), ("validation", val_loader), ("test", test_loader)]:
        if loader is None or len(loader.dataset) == 0:
            print(f"Skipping {split_name}: no data")
            continue

        metrics = evaluate_model(model, loader, criterion, device, num_classes)
        results[split_name] = metrics

        print(f"\n{split_name.upper()} Results:")
        print(f"  Loss:        {metrics['loss']:.4f}")
        print(f"  Accuracy:    {metrics['accuracy']:.4f}")
        print(f"  Precision:   {metrics['precision']:.4f}")
        print(f"  Recall:      {metrics['recall']:.4f}")
        print(f"  F1 Score:    {metrics['f1']:.4f}")
        print(f"  Specificity: {metrics['specificity']:.4f}")

    return results
