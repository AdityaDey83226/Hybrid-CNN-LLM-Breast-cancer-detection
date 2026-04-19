import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision.models import squeezenet1_0, SqueezeNet1_0_Weights
from prepare_data import create_dataloaders
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_curve, auc, log_loss,
                             balanced_accuracy_score, matthews_corrcoef, cohen_kappa_score)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


PLOT_DIR = "SQUEEZENET_ONLY_RESULTS"
if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)


CONFIG = {
    "40x": {"model": "",#example-squeezenet_40x.pth
            "data_path": r""},
    "100x": {"model": "",#example-squeezenet_100x.pth
             "data_path": r""},
    "200x": {"model": "",#example-squeezenet_200x.pth
             "data_path": r""},
    "400x": {"model": "",#example-squeezenet_400x.pth
             "data_path": r""}
}

all_true = []
cnn_all_pred = []
cnn_probs = []


for mag in CONFIG:
    print(f"Evaluating Baseline SqueezeNet 1.0 at: {mag}...")


    weights = SqueezeNet1_0_Weights.DEFAULT
    model = squeezenet1_0(weights=weights)

  
    model.classifier[1] = nn.Conv2d(512, 1, kernel_size=(1, 1))
    model.num_classes = 1

    # Load Trained Weights
    try:
        model.load_state_dict(torch.load(CONFIG[mag]["model"], map_location=device))
        model = model.to(device).eval()
    except FileNotFoundError:
        print(f"Skipping {mag}: Model file '{CONFIG[mag]['model']}' not found.")
        continue

    _, _, test_loader = create_dataloaders(CONFIG[mag]["data_path"])

    with torch.no_grad():
        for batch in test_loader:
            images, labels = batch[0].to(device), batch[1]

            outputs = model(images)
           
            outputs = outputs.view(outputs.size(0), -1)

            probs = torch.sigmoid(outputs).cpu().numpy().flatten()

            for i in range(len(probs)):
                p = probs[i]
                all_true.append(int(labels[i].item()))
                cnn_all_pred.append(int(p >= 0.5))
                cnn_probs.append(p)



def compute_metrics(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

  
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall (Sensitivity)": recall_score(y_true, y_pred),
        "Specificity": spec,
        "F1-Score": f1_score(y_true, y_pred),
        "Balanced Acc": balanced_accuracy_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "Cohen's Kappa": cohen_kappa_score(y_true, y_pred),
        "Log Loss (BCE)": log_loss(y_true, y_prob),
        "AUC-ROC": auc(*roc_curve(y_true, y_prob)[:2]),
        "False Positive Rate": fp / (fp + tn) if (fp + tn) > 0 else 0,
        "False Negative Rate": fn / (fn + tp) if (fn + tp) > 0 else 0
    }
    return metrics, cm


results_m, results_cm = compute_metrics(all_true, cnn_all_pred, cnn_probs)


plt.figure(figsize=(6, 5))
sns.heatmap(results_cm, annot=True, fmt='d', cmap='Purples', cbar=False)
plt.title('Baseline SqueezeNet: Global Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig(os.path.join(PLOT_DIR, "squeezenet_baseline_cm.png"), dpi=300)
plt.close()

# ROC Curve
fpr, tpr, _ = roc_curve(all_true, cnn_probs)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='indigo', lw=2, label=f'AUC = {auc(fpr, tpr):.4f}')
plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
plt.title('Baseline SqueezeNet: ROC Curve')
plt.legend(loc="lower right")
plt.savefig(os.path.join(PLOT_DIR, "squeezenet_baseline_roc.png"), dpi=300)
plt.close()


print("\n" + "=" * 50)
print("SQUEEZENET 1.0 BASELINE-ONLY EVALUATION")
print("=" * 50)
for metric, value in results_m.items():
    print(f"{metric:25}: {value:.4f}")

print(f"\nSUCCESS, Baseline metrics computed. Graphics saved to '{PLOT_DIR}'.")
