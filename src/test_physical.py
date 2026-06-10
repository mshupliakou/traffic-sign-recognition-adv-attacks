import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import random

# === 1. CONFIGURATION ===
DATASET_DIR = "../physical_test_images"
TARGET_ATTACK_CLASS = 1
MAX_IMAGES_TO_SHOW = 5  # Number of incorrect predictions to display

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FOLDER_TO_CLASS = {
    "30": 1,
    "50": 2,
    "80": 5,
    "blue_arrow": 38,
    "stop": 14,
    "triangle": 13,
    "minus": 17
}

# Reverse mapping to get readable names for the plots
CLASS_TO_FOLDER = {v: k for k, v in FOLDER_TO_CLASS.items()}

# === 2. MODEL LOADING ===
print("Loading trained ResNet18 model...")
model = resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 43)

model_path = "../models/gtsrb_resnet18.pth"
if not os.path.exists(model_path):
    model_path = "./models/gtsrb_resnet18.pth"

model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# === 3. IMAGE TRANSFORMATIONS ===
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])


# === 4. VISUALIZATION FUNCTION ===
def show_failed_examples(failed_list, condition_name):
    if not failed_list:
        return

    random.shuffle(failed_list)
    examples_to_show = failed_list[:MAX_IMAGES_TO_SHOW]

    fig, axes = plt.subplots(1, len(examples_to_show), figsize=(15, 4))
    if len(examples_to_show) == 1:
        axes = [axes]

    fig.suptitle(f"Misclassified Images: {condition_name}", fontsize=16, fontweight='bold')

    for ax, (img_path, true_class, pred_class) in zip(axes, examples_to_show):
        img = Image.open(img_path)
        ax.imshow(img)

        true_name = CLASS_TO_FOLDER.get(true_class, f"Class {true_class}")
        pred_name = CLASS_TO_FOLDER.get(pred_class, f"Class {pred_class}")

        ax.set_title(f"True: {true_name}\nPred: {pred_name}", color="red", fontsize=12)
        ax.axis('off')

    plt.tight_layout()
    plt.show()


# === 5. EVALUATION FUNCTION ===
def evaluate_folder(folder_path, is_patched=False):
    if not os.path.exists(folder_path):
        print(f"Directory not found: {folder_path}")
        return

    total_images = 0
    correct_predictions = 0
    targeted_success = 0
    untargeted_success = 0

    # List to store mistakes: tuples of (image_path, true_class, predicted_class)
    failed_examples = []

    for folder_name in os.listdir(folder_path):
        class_path = os.path.join(folder_path, folder_name)
        if not os.path.isdir(class_path):
            continue

        if folder_name not in FOLDER_TO_CLASS:
            continue

        true_class = FOLDER_TO_CLASS[folder_name]

        for img_name in os.listdir(class_path):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            img_path = os.path.join(class_path, img_name)
            total_images += 1

            try:
                image = Image.open(img_path).convert('RGB')
            except Exception:
                continue

            img_tensor = transform(image).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                output = model(img_tensor)
                pred = output.argmax(1).item()

            if pred == true_class:
                correct_predictions += 1
            else:
                # Store the failure for visualization
                failed_examples.append((img_path, true_class, pred))

            if is_patched:
                if pred == TARGET_ATTACK_CLASS:
                    targeted_success += 1
                if pred != true_class:
                    untargeted_success += 1

    condition_name = "PATCHED (ATTACK)" if is_patched else "CLEAN (BASELINE)"
    print(f"\n--- RESULTS FOR {condition_name} IMAGES ---")
    print(f"Total images evaluated: {total_images}")

    if total_images == 0:
        print("No images found to evaluate.")
        return

    if not is_patched:
        accuracy = (correct_predictions / total_images) * 100
        print(f"Baseline Accuracy: {accuracy:.2f}% (Model recognized the original sign)")
    else:
        targeted_asr = (targeted_success / total_images) * 100
        untargeted_asr = (untargeted_success / total_images) * 100
        model_resilience = (correct_predictions / total_images) * 100

        print(f"Targeted Attack Success Rate (Class {TARGET_ATTACK_CLASS}): {targeted_asr:.2f}%")
        print(f"Untargeted Attack Success Rate (Failed to recognize true class): {untargeted_asr:.2f}%")
        print(f"Model Resilience: {model_resilience:.2f}%")

    # Show the misclassified images in a pop-up window
    if failed_examples:
        print(f"Found {len(failed_examples)} misclassified images. Showing random examples on screen...")
        show_failed_examples(failed_examples, condition_name)


# === 6. EXECUTION ===
print("\n" + "=" * 50)
print("PHYSICAL ADVERSARIAL ATTACK EVALUATION REPORT")
print("=" * 50)

clean_dir = os.path.join(DATASET_DIR, "signs")
patched_dir = os.path.join(DATASET_DIR, "signs_with_patches")

evaluate_folder(clean_dir, is_patched=False)
evaluate_folder(patched_dir, is_patched=True)

print("\n" + "=" * 50)