# 🧠 Hybrid CNN–LLM Framework for Breast Cancer Classification

A lightweight and efficient deep learning framework that combines **MobileNetV2**, **entropy-based uncertainty estimation**, and **Large Language Model (LLM) reasoning** to improve diagnostic accuracy in breast cancer classification.

---

## 📌 Overview

Breast cancer classification from histopathological images is a challenging task due to complex tissue structures and ambiguous patterns. Traditional CNN-based approaches often produce **overconfident predictions**, especially in borderline cases.

This project proposes a **hybrid framework** that:
- Uses **lightweight CNNs** for efficient baseline classification  
- Applies **entropy-based uncertainty estimation** to detect ambiguous cases  
- Utilizes **Grad-CAM** for explainability  
- Integrates **LLM-based reasoning** for refining uncertain predictions  

The result is a system that is both **computationally efficient** and **diagnostically reliable**.

---

## Key Features

-  Lightweight CNN models (MobileNetV2, MobileNetV3, ResNet18, ShuffleNet, SqueezeNet)
-  Entropy-based uncertainty detection
-  Grad-CAM based explainability
-  Descriptor-driven prompt generation
-  LLM reasoning (Nous Hermes, Phi-4, Mistral)
-  Hybrid decision-making pipeline
-  Multi-magnification analysis (40X, 100X, 200X, 400X)

---

##  Proposed Framework


🔗 Interactive Diagram: *([https://lucid.app/lucidchart/1518b5bf-635b-4546-a3b6-d4727ef62a40/edit?viewport_loc=-751%2C-223%2C2794%2C1459%2C0_0&invitationId=inv_92376d3f-0e06-4f1d-b573-9da7d45e9287])*

##  MobileNetV2-Architecture

🔗 Interactive Diagram: *([https://lucid.app/lucidchart/8d48509d-fcef-47e5-bb58-39139411241a/edit?viewport_loc=-354%2C70%2C2794%2C1473%2C0_0&invitationId=inv_72b84de5-3ade-41cf-a66d-45bbc2ab5ef8])*
