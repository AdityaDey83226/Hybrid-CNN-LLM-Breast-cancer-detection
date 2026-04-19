import json
import requests
import pandas as pd
import time



OPENROUTER_API_KEY = ""  ##Enter API keyb here

API_URL = "" # enetr the url link of model

MODEL = "mistralai/mixtral-8x7b-instruct"    #Enter model name here


FILES = {
    "40x": "prompts_40x.json",
    "100x": "prompts_100x.json",
    "200x": "prompts_200x.json",
    "400x": "prompts_400x.json"
}


def query_llm(original_prompt, cnn_prob, entropy):

    system_prompt = f"""
You are a medical AI assistant analyzing histopathology descriptors for breast cancer classification.

You are given:
- Visual descriptors extracted from the image
- CNN probability (supporting information)
- Entropy (uncertainty level)

Your task:
Decide whether the tissue is BENIGN or MALIGNANT.

Guidelines:
- Use descriptors as primary evidence
- Use CNN probability as supporting signal
- Be balanced: do NOT always favor benign
- If descriptors show irregularity, heterogeneity, or fragmentation → consider malignant
- If descriptors show uniformity and localization → consider benign

---

CNN Probability: {round(cnn_prob, 3)}
Entropy: {round(entropy, 3)}

{original_prompt}

---

Respond STRICTLY in this format:

Diagnosis: <Benign or Malignant>
Confidence: <Low / Medium / High>
Explanation: <Short reasoning>
"""

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt}
        ],
        "temperature": 0.3,   # slightly higher → less bias
        "max_tokens": 300
    }

    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code != 200:
        print("API ERROR:", response.status_code)
        print(response.text)
        return "Diagnosis: Unknown\nConfidence: Low\nExplanation: API Error"

    try:
        result = response.json()
        text = result["choices"][0]["message"]["content"]
    except:
        print("Invalid response:", response.text)
        return "Diagnosis: Unknown\nConfidence: Low\nExplanation: Invalid API response"

    return text

# ------------------------------------------------
# PARSE LLM OUTPUT
# ------------------------------------------------

def parse_response(text):

    diagnosis = "Unknown"
    confidence = "Unknown"
    explanation = text

    text_lower = text.lower()

    if "malignant" in text_lower:
        diagnosis = "Malignant"

    if "benign" in text_lower:
        diagnosis = "Benign"

    if "high" in text_lower:
        confidence = "High"

    elif "medium" in text_lower:
        confidence = "Medium"

    elif "low" in text_lower:
        confidence = "Low"

    return diagnosis, confidence, explanation


# ------------------------------------------------
# MAIN LOOP
# ------------------------------------------------

records = []

for mag in FILES:

    print("\nProcessing:", FILES[mag])

    with open(FILES[mag], "r") as f:
        prompts = json.load(f)

    for item in prompts:
        prompt = item["prompt"]

        response_text = query_llm(
            prompt,
            item["probability"],
            item["entropy"]
        )

        diagnosis, confidence, explanation = parse_response(response_text)

        records.append({

            "image_path": item["image_path"],
            "magnification": mag,
            "cnn_probability": item["probability"],
            "entropy": item["entropy"],
            "descriptors": item["descriptors"],
            "llm_diagnosis": diagnosis,
            "llm_confidence": confidence,
            "llm_explanation": explanation

        })

        print("Processed:", len(records))

        time.sleep(1)  



df = pd.DataFrame(records)

df.to_csv("llm_predictions_MISTRAL.csv", index=False)

print("\nLLM reasoning complete")
print("Saved: llm_predictions_MISTRAL.csv")
print("Total processed:", len(records))
