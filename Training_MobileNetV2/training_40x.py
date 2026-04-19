
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models import MobileNet_V2_Weights
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from prepare_data import create_dataloaders



MAG_PATH = r""
MODEL_NAME = "mobilenet_40x.pth"


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load data (test_loader ignored during training)
    train_loader, val_loader, _ = create_dataloaders(MAG_PATH)

    # ==========================================
    # CLASS WEIGHT (FOR IMBALANCE)
    # ==========================================
    labels_np = train_loader.dataset.df["label"].values
    num_pos = np.sum(labels_np == 1)
    num_neg = np.sum(labels_np == 0)

    pos_weight_value = num_neg / num_pos
    print("Positive class weight:", pos_weight_value)

    pos_weight = torch.tensor([pos_weight_value]).to(device)

    # ==========================================
    # MODEL SETUP
    # ==========================================
    weights = MobileNet_V2_Weights.DEFAULT
    model = models.mobilenet_v2(weights=weights)

    # Freeze early layers (reduce overfitting + faster training)
    for param in model.features[:10].parameters():
        param.requires_grad = False

    # Replace classifier
    model.classifier[1] = nn.Linear(model.last_channel, 1)

    model = model.to(device)


    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    EPOCHS = 15
    best_val_f1 = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "train_f1": [],
        "val_f1": []
    }

    # TRAINING LOOP
 
    for epoch in range(EPOCHS):

        # -------- TRAIN --------
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.unsqueeze(1).to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            probs = torch.sigmoid(outputs).detach().cpu().numpy()
            preds = (probs > 0.5).astype(int)

            train_preds.extend(preds.flatten())
            train_labels.extend(labels.cpu().numpy().flatten())

        train_loss /= len(train_loader)

        train_acc = accuracy_score(train_labels, train_preds)
        train_prec = precision_score(train_labels, train_preds)
        train_rec = recall_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds)

        # -------- VALIDATION --------
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for images, labels in val_loader:

                images = images.to(device)
                labels = labels.unsqueeze(1).to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                probs = torch.sigmoid(outputs).cpu().numpy()
                preds = (probs > 0.5).astype(int)

                val_preds.extend(preds.flatten())
                val_labels.extend(labels.cpu().numpy().flatten())

        val_loss /= len(val_loader)

        val_acc = accuracy_score(val_labels, val_preds)
        val_prec = precision_score(val_labels, val_preds)
        val_rec = recall_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds)

        # Store history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_f1"].append(train_f1)
        history["val_f1"].append(val_f1)

        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        print(f"Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f}")

        # Save best model based on validation F1
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), MODEL_NAME)
            print("Best model saved.")

    print("\nTraining Complete.")


if __name__ == "__main__":
    main()
