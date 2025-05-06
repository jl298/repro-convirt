from torchvision import transforms

def get_transforms_pretrain():
    # Training image transformation (including data augmentation)
    # - Random crop (ratio 0.6-1.0)
    # - Horizontal flip (p=0.5)
    # - Affine transformation (rotation ±20 degrees, translation 0.1, scale 0.95-1.05)
    # - Color jittering (brightness/contrast only 0.6-1.4)
    # - Gaussian blur (σ=0.1-3.0)
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.95, 1.05)),
        transforms.ColorJitter(brightness=(0.6 / 1.4, 1.4 / 0.6), contrast=(0.6 / 1.4, 1.4 / 0.6)),
        transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 3.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Validation/test image transformation (no data augmentation)
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform

def get_transforms_classification():
    class SquarePad:
        def __call__(self, image):
            w, h = image.size
            max_wh = max(w, h)
            hp = int((max_wh - w) / 2)
            vp = int((max_wh - h) / 2)
            padding = (hp, vp, hp + (max_wh - w) % 2, vp + (max_wh - h) % 2)
            return transforms.functional.pad(image, padding, 0, 'constant')

    # Training image transformation (including data augmentation)
    # - First zero-pad to square
    # - Random crop (ratio 0.6-1.0)
    # - Horizontal flip (p=0.5)
    # - Affine transformation (rotation ±20 degrees, translation 0.1, scale 0.95-1.05)
    # - Color jittering (brightness/contrast only 0.6-1.4)
    # - Gaussian blur (σ=0.1-3.0)
    train_transform = transforms.Compose([
        SquarePad(),
        transforms.Resize(224),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.95, 1.05)),
        transforms.ColorJitter(brightness=(0.6 / 1.4, 1.4 / 0.6), contrast=(0.6 / 1.4, 1.4 / 0.6)),
        transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 3.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Validation/test image transformation (no data augmentation)
    # First zero-pad to square, then resize to 224x224
    val_transform = transforms.Compose([
        SquarePad(),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform
