import os
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms



def build_dataframe(root_path):
    data = []

    for label_name in ["benign", "malignant"]:
        label_path = os.path.join(root_path, label_name)

        if not os.path.exists(label_path):
            continue

        for img_name in os.listdir(label_path):
            img_path = os.path.join(label_path, img_name)

            try:
                # Extract patient ID (BreaKHis format)
                patient_id = img_name.split('-')[2]
            except:
                patient_id = img_name  # fallback

            label = 0 if label_name == "benign" else 1

            data.append({
                "filepath": img_path,
                "label": label,
                "patient_id": patient_id
            })

    return pd.DataFrame(data)



def patient_split(df, seed=42):
    patients = df["patient_id"].unique()

    train_patients, temp_patients = train_test_split(
        patients, test_size=0.30, random_state=seed
    )

    val_patients, test_patients = train_test_split(
        temp_patients, test_size=0.50, random_state=seed
    )

    train_df = df[df["patient_id"].isin(train_patients)]
    val_df = df[df["patient_id"].isin(val_patients)]
    test_df = df[df["patient_id"].isin(test_patients)]

    return train_df, val_df, test_df



class BreakHisDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, "filepath"]
        label = self.df.loc[idx, "label"]

        try:
            image = Image.open(img_path).convert("RGB")
        except:
            # Handle corrupted image
            return None

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32), img_path




def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])



def create_dataloaders(root_path, batch_size=8):

    df = build_dataframe(root_path)

    if df.empty:
        raise ValueError("Dataset is empty. Check your root_path.")

    train_df, val_df, test_df = patient_split(df)

    train_dataset = BreakHisDataset(train_df, transform=get_transforms(train=True))
    val_dataset = BreakHisDataset(val_df, transform=get_transforms(train=False))
    test_dataset = BreakHisDataset(test_df, transform=get_transforms(train=False))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


def print_dataset_info(root_path):
    print("\nExpected dataset structure:")
    print("root_path/")
    print("  ├── benign/")
    print("  ├── malignant/")
    print("\nGiven path:", root_path)
