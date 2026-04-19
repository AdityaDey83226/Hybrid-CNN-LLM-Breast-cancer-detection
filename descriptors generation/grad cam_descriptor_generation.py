import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import cv2
import os
from torchvision import models, transforms
from torchvision.models import MobileNet_V2_Weights
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



CONFIG = {
    "40x": {
        "model": "",#Example-mobilenet_40x.pth
        "edge_csv": "edge cases_40x.csv"
    },
    "100x": {
        "model": "",#Example-mobilenet_100x.pth
        "edge_csv": "edge cases_100x.csv"
    },
    "200x": {
        "model": "",#Example-mobilenet_200x.pth
        "edge_csv": "edge cases_200x.csv"
    },
    "400x": {
        "model": "",#Example-mobilenet_400x.pth
        "edge_csv": "edge cases_400x.csv"
    }
}

OUTPUT_DIR = "gradcam_maps_test"
os.makedirs(OUTPUT_DIR, exist_ok=True)


transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])


def extract_descriptors(cam):

    activation_area = np.mean(cam > 0.5)
    intensity = np.mean(cam)
    dispersion = np.std(cam)

    binary = (cam > 0.5).astype(np.uint8)
    num_labels, _ = cv2.connectedComponents(binary)

    descriptors = []

    if intensity > 0.5:
        descriptors.append("High activation intensity")
    else:
        descriptors.append("Moderate activation intensity")

    if activation_area > 0.3:
        descriptors.append("Widespread activation regions")
    else:
        descriptors.append("Localized activation clusters")

    if dispersion > 0.25:
        descriptors.append("High structural heterogeneity")
    else:
        descriptors.append("Moderate structural uniformity")

    if num_labels > 3:
        descriptors.append("Fragmented activation pattern")
    else:
        descriptors.append("Compact activation structure")

    return "; ".join(descriptors)


for mag in CONFIG:

    print(f"\nProcessing {mag}...")

    model_path = CONFIG[mag]["model"]
    edge_csv = CONFIG[mag]["edge_csv"]

    df = pd.read_csv(edge_csv)

    # Load model
    weights = MobileNet_V2_Weights.DEFAULT
    model = models.mobilenet_v2(weights=weights)

    model.classifier[1] = nn.Linear(model.last_channel,1)
    model.load_state_dict(torch.load(model_path, map_location=device))

    model = model.to(device)
    model.eval()

    gradients = []
    activations = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    target_layer = model.features[-1]
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_full_backward_hook(backward_hook)

    mag_dir = os.path.join(OUTPUT_DIR, mag)
    os.makedirs(mag_dir, exist_ok=True)

    descriptor_records = []

    for idx, row in df.iterrows():

        image_path = row["image_path"]

        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        model.zero_grad()

        output = model(image_tensor)
        prob = torch.sigmoid(output)

        output[0].backward()

        grads = gradients[-1]
        acts = activations[-1]

        weights_cam = torch.mean(grads, dim=(2,3), keepdim=True)
        cam = torch.sum(weights_cam * acts, dim=1).squeeze()

        cam = torch.relu(cam)
        cam = cam.detach().cpu().numpy()

        cam = cv2.resize(cam, (224,224))
        cam = cam / (cam.max() + 1e-8)

        heatmap = np.uint8(255 * cam)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        heatmap_path = os.path.join(mag_dir, f"heatmap_{idx}.png")
        cv2.imwrite(heatmap_path, heatmap)

        descriptors = extract_descriptors(cam)

        descriptor_records.append({
            "image_path": image_path,
            "prob_malignant": row["prob_malignant"],
            "entropy": row["entropy"],
            "descriptors": descriptors,
            "magnification": mag
        })

    out_csv = f"descriptors_test_{mag}.csv"
    pd.DataFrame(descriptor_records).to_csv(out_csv, index=False)

    print(f"Saved descriptors: {out_csv}")
    print("Total descriptors:", len(descriptor_records))

print("\nGrad-CAM descriptor generation complete."
