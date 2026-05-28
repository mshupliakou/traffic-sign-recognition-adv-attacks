from torchvision.datasets import GTSRB
from torchvision import transforms
from torch.utils.data import random_split

IMG_SIZE = 64

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

print("Downloading GTSRB dataset...")

train_full = GTSRB(
    root="./data",
    split="train",
    download=True,
    transform=transform,
)

test_dataset = GTSRB(
    root="./data",
    split="test",
    download=True,
    transform=transform,
)

train_size = int(0.8 * len(train_full))
val_size = len(train_full) - train_size

train_dataset, val_dataset = random_split(
    train_full,
    [train_size, val_size]
)

print()
print("Dataset ready")
print(f"Train samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")
