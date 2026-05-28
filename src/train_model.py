import torch
import torch.nn as nn
import torch.optim as optim
import os

from torchvision.datasets import GTSRB
from torchvision import transforms
from torchvision.models import (
    resnet18,
    ResNet18_Weights,
)

from torch.utils.data import DataLoader
from torch.utils.data import random_split

from tqdm import tqdm

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 64
IMG_SIZE = 64
EPOCHS = 5
LR = 1e-4

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

print("Loading dataset...")

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

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

print("Loading pretrained ResNet18...")

model = resnet18(
    weights=ResNet18_Weights.DEFAULT
)

model.fc = nn.Linear(
    model.fc.in_features,
    43
)

model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LR,
)

def evaluate(loader):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            preds = outputs.argmax(1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

    return 100 * correct / total

print("Training...")

best_acc = 0

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0

    for images, labels in tqdm(train_loader):

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    val_acc = evaluate(val_loader)

    print()
    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"Loss: {running_loss:.4f}")
    print(f"Validation accuracy: {val_acc:.2f}%")

    if val_acc > best_acc:

        best_acc = val_acc
        os.makedirs('./models', exist_ok=True)
        torch.save(
            model.state_dict(),
            "./models/gtsrb_resnet18.pth"
        )

        print("Model saved")

print()
print("Training finished")
