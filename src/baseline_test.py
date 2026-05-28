import torch
import torch.nn as nn

from torchvision.datasets import GTSRB
from torchvision import transforms
from torchvision.models import resnet18

from torch.utils.data import DataLoader

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

IMG_SIZE = 64
BATCH_SIZE = 32

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

print("Loading test dataset...")

test_dataset = GTSRB(
    root="./data",
    split="test",
    download=True,
    transform=transform,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

print("Loading model...")

model = resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    43
)

model.load_state_dict(
    torch.load(
        "./models/gtsrb_resnet18.pth",
        map_location=DEVICE,
    )
)

model = model.to(DEVICE)

model.eval()

print()
print("Running baseline evaluation...")
print()

correct = 0
total = 0

samples_to_show = 15
shown = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        preds = outputs.argmax(1)

        total += labels.size(0)

        correct += (preds == labels).sum().item()

        for i in range(len(images)):

            if shown >= samples_to_show:
                break

            gt = labels[i].item()
            pred = preds[i].item()

            status = "OK" if gt == pred else "FAIL"

            print(
                f"Sample {shown+1:02d} | "
                f"GT: {gt:02d} | "
                f"PRED: {pred:02d} | "
                f"{status}"
            )

            shown += 1

        if shown >= samples_to_show:
            break

accuracy = 100 * correct / total

print()
print("=" * 50)
print(f"Baseline accuracy: {accuracy:.2f}%")
print("=" * 50)
