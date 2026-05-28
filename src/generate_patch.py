import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from art.attacks.evasion import AdversarialPatchPyTorch
from art.estimators.classification import PyTorchClassifier

from torchvision.models import resnet18
from torchvision.datasets import GTSRB
from torchvision import transforms

# 1. CHOSEN SIGNS CLASSES
# Target sign IDs for the attack
TARGET_CLASSES = [14, 17, 13, 1, 38]
# Target attack class (e.g., 1 - Speed limit 30)
TARGET_ATTACK_CLASS = 1

# Setup device (GPU or CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. REAL MODEL
print("Loading trained model...")
# Load the same architecture used by the partner
model = resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 43)

# Load saved weights (model should be located in the 'models' directory)
model.load_state_dict(torch.load("./models/gtsrb_resnet18.pth", map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# 3. ART LIBRARY CONFIGURATION
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

classifier = PyTorchClassifier(
    model=model,
    clip_values=(0.0, 1.0),
    loss=criterion,
    optimizer=optimizer,
    input_shape=(3, 64, 64), # UPDATED: partner used 64x64 input size
    nb_classes=43,
    device_type="gpu" if torch.cuda.is_available() else "cpu"
)


# 4. ATTACK CONFIGURATION
print("Configuring adversarial patch parameters...")
attack = AdversarialPatchPyTorch(
    estimator=classifier,
    rotation_max=22.5,
    scale_min=0.1,
    scale_max=0.4,
    learning_rate=0.05,
    max_iter=500,
    batch_size=16,
    targeted=True
)

# 5. PREPARING REAL IMAGES
print("Loading real images from GTSRB dataset...")
IMG_SIZE = 64
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# Load test dataset
test_dataset = GTSRB(root="./data", split="test", download=True, transform=transform)

real_images = []
labels_collected = []

# Find one image for each of our 5 chosen signs
for img, label in test_dataset:
    if label in TARGET_CLASSES and label not in labels_collected:
        real_images.append(img.numpy())
        labels_collected.append(label)
    # Stop searching if we found all 5 signs
    if len(real_images) == len(TARGET_CLASSES):
        break

real_images = np.array(real_images)

target_labels = np.zeros((len(TARGET_CLASSES), 43), dtype=np.float32)
for i in range(len(TARGET_CLASSES)):
    target_labels[i][TARGET_ATTACK_CLASS] = 1.0

# 6. PATCH GENERATION
print("Generating patch... (Mathematical optimization)")
patch, patch_mask = attack.generate(x=real_images, y=target_labels)

print("Generation completed successfully!")

# 7. SAVING THE RESULT
plt.imshow(np.transpose(patch, (1, 2, 0)))
plt.title("Adversarial Patch (Digital)")
plt.axis('off')
plt.savefig('my_adversarial_patch.png')
print("Patch saved to file 'my_adversarial_patch.png'")

# === 8. VISUALIZING THE ATTACK ===
print("Applying patch to real images and evaluating...")

# Predict classes BEFORE the attack
preds_before = classifier.predict(real_images)
classes_before = np.argmax(preds_before, axis=1)

# Apply the generated patch to the real images
patched_images = attack.apply_patch(real_images, scale=0.3)

# Predict classes AFTER the attack
preds_after = classifier.predict(patched_images)
classes_after = np.argmax(preds_after, axis=1)

# Visualization Setup
fig, axes = plt.subplots(2, len(real_images), figsize=(15, 6))
fig.suptitle("Adversarial Patch Attack: Before vs After", fontsize=16)

for i in range(len(real_images)):
    # 1. Top row: Original Images
    ax_orig = axes[0, i]
    # Convert from PyTorch format (C, H, W) to Matplotlib format (H, W, C)
    img_orig = np.transpose(real_images[i], (1, 2, 0))
    ax_orig.imshow(img_orig)
    ax_orig.set_title(f"Original\nPred: {classes_before[i]}\nTrue: {labels_collected[i]}")
    ax_orig.axis('off')

    # 2. Bottom row: Patched Images
    ax_patched = axes[1, i]
    img_patched = np.transpose(patched_images[i], (1, 2, 0))
    ax_patched.imshow(img_patched)

    # Check if the attack was successful
    success = (classes_after[i] == TARGET_ATTACK_CLASS)
    color = "green" if success else "red"

    ax_patched.set_title(f"Patched\nPred: {classes_after[i]}", color=color)
    ax_patched.axis('off')

plt.tight_layout()
plt.savefig('attack_results.png')
print("Visual results saved to 'attack_results.png'")
# plt.show() # Uncomment this if you want an interactive window to pop up