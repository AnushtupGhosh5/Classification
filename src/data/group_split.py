"""Deterministic group-aware stratification without external dependencies."""

import random
from collections import Counter, defaultdict


def greedy_group_stratified_split(records, num_classes, fractions, seed=42):
    """Split ``(path, label, group_id)`` records without group leakage.

    Groups are ordered by class rarity, then assigned to the partition that
    produces the smallest weighted deviation from the requested per-class and
    total-image targets. This supports patients containing more than one
    diagnosis, which PAD-UFES-20 does.
    """
    if len(fractions) < 2 or any(fraction <= 0 for fraction in fractions):
        raise ValueError("Split fractions must contain at least two positive values")
    if abs(sum(fractions) - 1.0) > 1e-8:
        raise ValueError(f"Split fractions must sum to 1, got {sum(fractions)}")
    if not records:
        raise ValueError("Cannot split an empty record list")

    grouped = defaultdict(list)
    totals = [0] * num_classes
    for path, label, group_id in records:
        if label < 0 or label >= num_classes:
            raise ValueError(f"Label {label} is outside [0, {num_classes})")
        grouped[group_id].append((path, label, group_id))
        totals[label] += 1

    if any(total == 0 for total in totals):
        missing = [index for index, total in enumerate(totals) if total == 0]
        raise ValueError(f"Cannot stratify because classes {missing} have no samples")

    targets = [
        [fraction * total for total in totals] for fraction in fractions
    ]
    size_targets = [fraction * len(records) for fraction in fractions]
    class_counts = [[0] * num_classes for _ in fractions]
    split_sizes = [0] * len(fractions)
    partitions = [[] for _ in fractions]

    rng = random.Random(seed)
    groups = list(grouped.items())
    rng.shuffle(groups)

    def rarity_key(item):
        counts = Counter(record[1] for record in item[1])
        shares = [count / totals[label] for label, count in counts.items()]
        return max(shares), sum(shares), len(item[1])

    groups.sort(key=rarity_key, reverse=True)

    for _group_id, group_records in groups:
        group_counts = Counter(record[1] for record in group_records)
        group_size = len(group_records)
        candidates = []
        for split_index in range(len(fractions)):
            before = sum(
                (class_counts[split_index][label] - targets[split_index][label]) ** 2
                / max(targets[split_index][label], 1.0)
                for label in range(num_classes)
            )
            after = sum(
                (
                    class_counts[split_index][label]
                    + group_counts[label]
                    - targets[split_index][label]
                ) ** 2
                / max(targets[split_index][label], 1.0)
                for label in range(num_classes)
            )
            before += 0.25 * (
                split_sizes[split_index] - size_targets[split_index]
            ) ** 2 / max(size_targets[split_index], 1.0)
            after += 0.25 * (
                split_sizes[split_index] + group_size - size_targets[split_index]
            ) ** 2 / max(size_targets[split_index], 1.0)
            candidates.append((after - before, split_index))

        _cost, selected = min(candidates)
        partitions[selected].extend(group_records)
        split_sizes[selected] += group_size
        for label, count in group_counts.items():
            class_counts[selected][label] += count

    group_sets = [
        {group_id for _path, _label, group_id in partition}
        for partition in partitions
    ]
    for left in range(len(group_sets)):
        for right in range(left + 1, len(group_sets)):
            if group_sets[left] & group_sets[right]:
                raise RuntimeError("Group-aware split unexpectedly contains overlap")
    for split_index, counts in enumerate(class_counts):
        missing = [label for label, count in enumerate(counts) if count == 0]
        if missing:
            raise RuntimeError(
                f"Split {split_index} is missing classes {missing}; choose another seed"
            )

    return partitions
