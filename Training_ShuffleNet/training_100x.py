
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from prepare_data import create_dataloaders


MAG_PATH = r""
MODEL_NAME = "shufflenet_100x.pth"


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, val_loader, _ = create_dataloaders(MAG_PATH)

    
    labels_np = train_loader.dataset.df["label"].values
    pos_weight = torch.tensor([np.sum(labels_np == 0) / np.sum(labels_np == 1)]).to(device)

    # Model
    weights = ShuffleNet_V2_X1_0_Weights.DEFAULT
    model = shufflenet_v2_x1_0(weights=weights)

    model.fc = nn.Linear(model.fc.in_features, 1)
    model = model.to(device)

  
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    EPOCHS = 20
    best_val_f1 = 0
    patience = 4
    counter = 0

    for epoch in range(EPOCHS):

        # TRAIN
        model.train()
        train_preds, train_labels = [], []
        train_loss = 0

        for images, labels, _ in train_loader:

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

        train_acc = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds)

    
        model.eval()
        val_preds, val_labels = [], []
        val_loss = 0

        with torch.no_grad():
            for images, labels, _ in val_loader:

                images = images.to(device)
                labels = labels.unsqueeze(1).to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                probs = torch.sigmoid(outputs).cpu().numpy()
                preds = (probs > 0.5).astype(int)

                val_preds.extend(preds.flatten())
                val_labels.extend(labels.cpu().numpy().flatten())

        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds)

        print(f"\nEpoch {epoch+1}")
        print(f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        print(f"Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), MODEL_NAME)
            print(" Best model saved")
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            print(" Early stopping")
            break

    print("\nTraining Complete.")


if __name__ == "__main__":
    main()
