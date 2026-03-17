import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

# =========================
# 1. LOAD DINOv2
# =========================
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
model.eval()

# =========================
# 2. TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# =========================
# 3. LABEL MAP
# =========================
label_map = {
    "noise": 0,
    "drone": 1,
    "drone-dist-noise-spectro": 1  
}

# =========================
# 4. DUYỆT DATASET
# =========================
X = []
y = []

dataset_path = "Spectrograms"

for class_name in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, class_name)

    if not os.path.isdir(class_path):
        continue

    label = label_map[class_name]

    for file in os.listdir(class_path):
        img_path = os.path.join(class_path, file)

        try:
            img = Image.open(img_path).convert("RGB")
            img = transform(img).unsqueeze(0)

            with torch.no_grad():
                emb = model(img).numpy().squeeze()

            X.append(emb)
            y.append(label)

        except Exception as e:
            print("Error:", img_path)

# =========================
# 5. CONVERT → NUMPY
# =========================
X = np.array(X)
y = np.array(y)

print("X shape:", X.shape)
print("y shape:", y.shape)

# =========================
# 6. SAVE
# =========================
np.save("X.npy", X)
np.save("y.npy", y)

print("Saved X.npy và y.npy")