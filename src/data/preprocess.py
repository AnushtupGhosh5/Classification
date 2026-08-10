from torchvision import transforms
import random
import torch


# ImageNet normalization (matches PyTorch pretrained backbone statistics).
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


class RandomRightAngleRotation:
    """Orientation augmentation without interpolated black corner artifacts."""

    def __call__(self, image):
        return image.rotate(random.choice((0, 90, 180, 270)))


class HorizontalThreeCrop:
    """Return left, centre, and right square crops from a landscape image."""

    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        width, height = image.size
        top = max((height - self.size) // 2, 0)
        offsets = (0, max((width - self.size) // 2, 0), max(width - self.size, 0))
        return tuple(
            image.crop((left, top, left + self.size, top + self.size))
            for left in offsets
        )


def _stack_normalized_crops(crops):
    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])
    return torch.stack([normalize(crop) for crop in crops])


def get_train_transforms(img_size=224, augment=True, augment_style="balanced"):
    """Build the training transform.

    ``balanced`` keeps the previous conservative crop/flip/color pipeline.
    ``seefnet`` mirrors the stronger reference notebooks: resize to the target
    square, then apply rotation/shift/shear/zoom plus horizontal flips. This is
    useful for ISIC/MILK10K where the base models were underperforming because
    they saw much less geometric variation than the reference setup.
    """
    scale = int(img_size * 256 / 224)  # e.g. 256 for img_size=224

    if not augment:
        return transforms.Compose([
            transforms.Resize((scale, scale)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])

    if augment_style == "seefnet":
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(
                degrees=25,
                translate=(0.25, 0.25),
                scale=(0.75, 1.25),
                shear=25,
                fill=0,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])

    if augment_style == "skin":
        # Dermatology images are close to square and the lesion is often the
        # only useful object.  Keep the lesion in frame and use small
        # photometric/geometric changes; large shear/translation can move it
        # out of frame or introduce black borders that are absent at test time.
        return transforms.Compose([
            transforms.Resize((scale, scale)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(180),
            transforms.ColorJitter(
                brightness=0.08, contrast=0.12, saturation=0.08, hue=0.02,
            ),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            transforms.RandomErasing(p=0.15, scale=(0.01, 0.08), ratio=(0.5, 2.0)),
        ])

    if augment_style == "skin_focus":
        # HAM10000/ISIC18 images are consistently 4:3 with lesions near the
        # centre. Resize the short edge and crop to square instead of warping
        # the lesion geometry from 600x450 directly into a square.
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            RandomRightAngleRotation(),
            transforms.ColorJitter(
                brightness=0.10, contrast=0.12, saturation=0.10, hue=0.02,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            transforms.RandomErasing(p=0.10, scale=(0.01, 0.05), ratio=(0.5, 2.0)),
        ])

    return transforms.Compose([
        transforms.Resize((scale, scale)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


def get_val_transforms(img_size=224, augment_style="balanced"):
    """Deterministic eval transform matched to the training resize policy."""
    if augment_style in ("seefnet", "skin"):
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])

    if augment_style == "skin_focus":
        return transforms.Compose([
            transforms.Resize(img_size),
            HorizontalThreeCrop(img_size),
            transforms.Lambda(_stack_normalized_crops),
        ])

    scale = int(img_size * 256 / 224)  # e.g. 256 for img_size=224
    return transforms.Compose([
        transforms.Resize((scale, scale)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])
