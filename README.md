
## Intelligent Heart Disease Risk Prediction System with LLM-Based Clinical Explanation
`2026-01-04`

### Overview

Cardiovascular diseases remain one of the leading causes of mortality worldwide. Early identification of heart disease risk can significantly reduce severe outcomes through timely medical intervention and lifestyle modification. However, many existing prediction systems either lack interpretability or are inaccessible to non-technical users.
This project presents a complete end-to-end machine learning–based Heart Disease Risk Prediction System integrated with a Large Language Model (LLM)–powered explanation engine, designed to assist patients and healthcare stakeholders in understanding risk factors clearly and intuitively.
The system not only predicts a patient’s heart disease risk level but also explains the prediction in human-readable language, bridging the gap between clinical ML outputs and patient comprehension.

### Problem Statement
Traditional ML prediction models often act as black boxes, providing numerical outputs without meaningful explanations. This limits trust, usability, and adoption—especially in healthcare scenarios involving patients with no medical or technical background.
```mermaid 
flowchart TB

subgraph UI["User Interface"]
    U1["User Browser"]
    U2["HTML Forms<br/>(Patient Data Input)"]
    U3["Results Dashboard<br/>(Risk + Probabilities)"]
    U4["AI Chat Interface<br/>(Lifestyle Guidance)"]

    U1 --> U2
    U2 --> U3
    U3 --> U4
end

subgraph WEB["Flask Web Application"]
    A1["serve.py<br/>(App Entrypoint)"]
    A2["Flask App Factory<br/>(create_app)"]
    A3["Routes Blueprint<br/>(routes.py)"]
    A4["Session Manager<br/>(Flask Session)"]

    A1 --> A2
    A2 --> A3
    A3 --> A4
end

subgraph ML["Prediction Service Layer"]
    S1["PredictionService"]
    S2["Feature Preprocessing<br/>(preprocessing.py)"]
    S3["Inference Client<br/>(Hugging Face REST API)"]

    S1 --> S2
    S1 --> S3
end

subgraph HF["Hugging Face Infrastructure"]
    H1["Hosted ML Model<br/>(Scikit-learn Pipeline)"]
    H2["Inference Endpoint"]
end

subgraph LLM["AI Explanation Engine"]
    G1["Gemini API"]
    G2["Contextual Health Advice"]
end

U2 --> A3
A3 --> S1
S3 --> H2
H2 --> H1
H1 --> S3
S1 --> A3
A3 --> U3

A3 --> G1
G1 --> G2
G2 --> A3
A3 --> U4
