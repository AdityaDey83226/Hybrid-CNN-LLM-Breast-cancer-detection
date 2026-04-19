import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torchvision import models
from torchvision.models import MobileNet_V2_Weights
from prepare_data import create_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##keeping threshold values through the empirical anaysis....
CONFIG = {
    "40x": {
        "model": "",##Example- mobilenet_40x.pth
        "data_path": r"",
        "entropy_threshold": 0.45
    },
    "100x": {
        "model": "",##Example-mobilenet_100x.pth
        "data_path": r"",
        "entropy_threshold": 0.60
    },
    "200x": {
        "model": "mobilenet_200x.pth",##Example-mobilenet_200x.pth
        "data_path": r"",
        "entropy_threshold": 0.45
    },
    "400x": {
        "model": "",##Example-mobilenet_400x.pth
        "data_path": r"",
        "entropy_threshold": 0.60
    }
}



def compute_entropy(p):

    return -p*np.log(p+1e-8)-(1-p)*np.log(1-p+1e-8)



for mag in CONFIG:

    print("\nProcessing",mag)

    model_path = CONFIG[mag]["model"]
    data_path = CONFIG[mag]["data_path"]
    threshold = CONFIG[mag]["entropy_threshold"]

    # load model
    weights = MobileNet_V2_Weights.DEFAULT
    model = models.mobilenet_v2(weights=weights)

    model.classifier[1] = nn.Linear(model.last_channel,1)
    model.load_state_dict(torch.load(model_path,map_location=device))

    model = model.to(device)
    model.eval()


    _,_,test_loader = create_dataloaders(data_path)

    dataset_df = test_loader.dataset.df.reset_index(drop=True)

    index_pointer = 0

    edge_records = []

    with torch.no_grad():

        for images,labels in test_loader:

            images = images.to(device)

            outputs = model(images)

            probs = torch.sigmoid(outputs).cpu().numpy().flatten()

            for i in range(len(probs)):

                p = probs[i]

                entropy = compute_entropy(p)

                if entropy >= threshold:

                    image_path = dataset_df.iloc[index_pointer]["filepath"]

                    edge_records.append({

                        "image_path":image_path,
                        "prob_malignant":p,
                        "entropy":entropy

                    })

                index_pointer += 1


    out_csv = f"edge_cases_test_{mag}.csv"

    pd.DataFrame(edge_records).to_csv(out_csv,index=False)

    print("Saved:",out_csv)
    print("Total edge cases:",len(edge_records))


print("\nTest edge case extraction complete.")
