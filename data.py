from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from cnn.config import DATA_DIR, IMG_SIZE, BATCH_SIZE, NUM_WORKERS, MEAN, STD

def cifar10_loaders(val_ratio: float = 0.1):
    """Create CIFAR-10 train/val/test DataLoaders with resize to 64."""
    train_tf = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    test_tf = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    train_full = datasets.CIFAR10(root=str(DATA_DIR), train=True,  download=True, transform=train_tf)
    test_set   = datasets.CIFAR10(root=str(DATA_DIR), train=False, download=True, transform=test_tf)

    val_size   = int(len(train_full) * val_ratio)
    train_size = len(train_full) - val_size
    train_set, val_set = random_split(train_full, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    return train_loader, val_loader, test_loader

