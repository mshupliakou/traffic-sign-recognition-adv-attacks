import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from art.attacks.evasion import AdversarialPatchPyTorch
from art.estimators.classification import PyTorchClassifier

#  1. CHOSEN SIGNS CLASSES
# Signs for attacking
TARGET_CLASSES = [14, 17, 13, 1, 38]
# Let's assume we want all these signs to be recognized as "Speed limit 30" (ID = 1)
TARGET_ATTACK_CLASS = 1

#  2. DUMMY MODEL
# Once Participant 1 provides the model, THIS PART SHOULD BE REMOVED...
class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3 * 32 * 32, 43) # 43 classes in GTSRB
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)

model = DummyModel()
model.eval()
# ...AND INSERT PARTICIPANT 1'S CODE HERE:
# model = RealGTSRBModel()
# model.load_state_dict(torch.load("model.pth"))

# 3. ART LIBRARY CONFIGURATION
# Wrapping the PyTorch model into the ART library format
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

classifier = PyTorchClassifier(
    model=model,
    clip_values=(0.0, 1.0), # Image normalization
    loss=criterion,
    optimizer=optimizer,
    input_shape=(3, 32, 32), # Standard image size in GTSRB
    nb_classes=43,
)

#  4. ATTACK CONFIGURATION (Adversarial Patch)
print("Configuring adversarial patch parameters...")
attack = AdversarialPatchPyTorch(
    estimator=classifier,
    rotation_max=22.5,  # Simulating a crookedly applied patch (rotation)
    scale_min=0.1,      # Minimum patch size (10% of the sign area)
    scale_max=0.3,      # Maximum patch size (30% of the area)
    learning_rate=0.01, # Patch learning rate
    max_iter=50,        # Number of iterations (can be increased to 500 later for better effect)
    batch_size=16
)

# 5. PATCH GENERATION
print("Preparing images for patch training...")
# INSTEAD OF THIS, you will need to load 5 real sign images (Participant 1 will write this)
dummy_images = np.random.rand(5, 3, 32, 32).astype(np.float32)
# Goal: force the model to think this is class 1 (Speed limit 30)
target_labels = np.array([TARGET_ATTACK_CLASS] * 5)

print("Generating patch... (Math optimization)")
# Generating the patch (passing the images and specifying the target incorrect class)
patch, patch_mask = attack.generate(x=dummy_images, y=target_labels)

print("Generation completed successfully!")

#  6. SAVING THE RESULT
# Saving the resulting "sticker" as an image
plt.imshow(np.transpose(patch, (1, 2, 0)))
plt.title("Adversarial Patch (Digital)")
plt.axis('off')
plt.savefig('my_adversarial_patch.png')
print("Patch saved to file 'my_adversarial_patch.png'")