import pandas as pd
import json


FILES = {
    "40x": "descriptors_40x.csv",
    "100x": "descriptors_100x.csv",
    "200x": "descriptors_200x.csv",
    "400x": "descriptors_400x.csv"
}


def build_prompt(descriptors, probability):

    descriptor_list = descriptors.split(";")

    formatted_descriptors = ""

    for d in descriptor_list:
        formatted_descriptors += "- " + d.strip() + "\n"

    prompt = (
        "Visual Findings:\n"
        + formatted_descriptors
        + "\nVision Model Probability of Malignancy: "
        + str(round(probability, 3))
        + "\n\nTask:\n"
        "Based on the visual descriptors and model probability, determine whether the breast tissue sample is BENIGN or MALIGNANT.\n\n"
        "Provide:\n"
        "1. Final diagnosis (Benign or Malignant)\n"
        "2. Confidence level (Low / Medium / High)\n"
        "3. Brief explanation based on the descriptors."
    )

    return prompt



for mag in FILES:

    file = FILES[mag]

    print("\nProcessing", mag)

    df = pd.read_csv(file)

    prompts = []

    for _, row in df.iterrows():

        prompt = build_prompt(
            row["descriptors"],
            row["prob_malignant"]
        )

        prompts.append({
            "image_path": row["image_path"],
            "magnification": mag,
            "probability": row["prob_malignant"],
            "entropy": row["entropy"],
            "descriptors": row["descriptors"],
            "prompt": prompt
        })

    output_file = "prompts_test_" + mag + ".json"

    with open(output_file, "w") as f:
        json.dump(prompts, f, indent=4)

    print("Saved:", output_file)
    print("Total prompts:", len(prompts))


print("\nPrompt generation for TEST set complete.")
