# Explainable-AI (XAI) in Deep Learning Models for Credit Card Fraud Detection

- **Run the live main notebook ([XAI_methods.ipynb](https://github.com/ThongLai/Credit-Card-Transaction-Fraud-Detection-Using-Explainable-AI/blob/main/XAI_methods.ipynb)):** [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ThongLai/Credit-Card-Transaction-Fraud-Detection-Using-Explainable-AI/main?urlpath=%2Fdoc%2Ftree%2FXAI_methods.ipynb)

- **Models: [architectures](https://github.com/ThongLai/Credit-Card-Transaction-Fraud-Detection-Using-Explainable-AI/tree/main/architectures)**<a name="models" id="models"></a>

## Description

This project gathers multiple state-of-the-art **Deep Learning models** used for credit card fraud detection and applies **Explainable AI (XAI)** techniques to enhance their interpretability and enable effective model comparison. We leverage synthetic data from the Sparkov dataset to train and evaluate our models.

**Key Features:**
- **Multi-Model Comparison:**  
  Assess and compare different deep learning architectures for fraud detection.
- **Explainable AI Integration:**  
  - **SHAP:** Interactive visualizations (force plots, summary plots, dependence plots) that elucidate feature contributions.  
  - **LIME:** Local interpretable explanations (in progress) to detail individual predictions.  
  - **Anchors:** Rule-based explanations offering high-precision insights.
- **Robust Evaluation:**  
  - Extensive testing on synthetic data with performance metrics to validate both model accuracy and interpretability.
  - Visual tools to explore feature impacts globally and locally, supporting transparent decision-making.

## System Architecture

The complete workflow includes:
1. **Data & Model Collection:**  
   Obtaining pre-trained fraud detection models and synthetic transaction data.
2. **Model Development Pipeline:**  
   Steps include data preprocessing, feature engineering, model training, evaluation, and hyperparameter tuning.
3. **Prediction & Comparison:**  
   Running inference to generate predictions and comparing performance across models.
4. **Explainability Analysis:**  
   Applying XAI methods (SHAP, LIME, Anchors) to reveal and compare how features drive model predictions.

```mermaid
---
config:
  theme: neo
---
flowchart TD
 subgraph Data_Preprocessing["<b>Data Preprocessing</b>"]
        A["Raw Transaction Data"]
        B["Feature Engineering"]
        C["Preprocessed Data"]
        D["Data Normalization"]
        E@{ label: "Class Balancing (<span style=\"--tw-scale-x:\"><b>SMOTE</b></span>)" }
  end
 subgraph Model_Training["<b>Model Training</b>"]
        F1["<b>CNN Pipeline</b>"]
        F2["<b>LSTM Pipeline</b>"]
        F1a["Feature Extraction"]
        F1b["CNN Model Training"]
        F1c["Model Optimization"]
        M1["Trained CNN Model"]
        F2a["Sequence Preparation"]
        F2b["LSTM Model Training"]
        F2c["Model Optimization"]
        M2["Trained LSTM Model"]
  end
 subgraph XAI_Integration["<b>XAI Integration</b>"]
        X1a["Apply <b>SHAP</b> Explanations"]
        X1b["Apply <b>LIME</b> Explanations"]
        X1c["Apply <b>Anchors</b> Explanations"]
        X3["Feature Importance Analysis"]
        X4["Local Explanation Generation"]
  end
 subgraph Performance_Evaluation["<b>XAI Metrics Evaluation</b>"]
        G1["<b>Faithfulness</b>"]
        G2["<b>Monotonicity</b>"]
        G3["<b>Completeness</b>"]
        Z["Visualisation"]
        Z1["XAI Evaluation Report"]
  end
    n2@{ label: "<b style=\"color:\">Sparkov</b> <span style=\"color:\">Dataset Acquisition</span>" } --> A
    A --> B
    B --> C
    C --> D
    D --> E
    E --> F1 & F2
    F1 --> F1a
    F1a --> F1b
    F1b --> F1c
    F1c --> M1
    F2 --> F2a
    F2a --> F2b
    F2b --> F2c
    F2c --> M2
    M1 --> X1a & X1b & X1c
    X1a --> X3 & X4
    X1b --> X3 & X4
    X1c --> X3 & X4
    G1 --> Z
    G2 --> Z
    G3 --> Z
    Z --> Z1
    M2 --> X1a & X1c & X1b
    Z1 --> n1["<b>Deployment Decision</b>"]
    X4 --> n1
    X3 --> G1 & G2 & G3
    D@{ shape: rect}
    E@{ shape: rect}
    n2@{ shape: rect}
     A:::preprocessing
     B:::preprocessing
     C:::preprocessing
     D:::preprocessing
     E:::preprocessing
     F1:::training
     F2:::training
     F1a:::training
     F1b:::training
     F1c:::training
     M1:::training
     F2a:::training
     F2b:::training
     F2c:::training
     M2:::training
     X1a:::xai
     X1b:::xai
     X1c:::xai
     X3:::xai
     X4:::xai
     G1:::evaluation
     G2:::evaluation
     G3:::evaluation
     Z:::evaluation
     Z1:::evaluation
     n2:::data
     n1:::decision
    classDef data fill:#b3e6ff,stroke:#35424a,stroke-width:3px,color:#222222
    classDef preprocessing fill:#ffcccb,stroke:#35424a,stroke-width:2px,color:#222222
    classDef training fill:#c4e17f,stroke:#35424a,stroke-width:2px,color:#222222
    classDef xai fill:#f3d1ff,stroke:#35424a,stroke-width:2px,color:#222222
    classDef evaluation fill:#ffeb99,stroke:#35424a,stroke-width:2px,color:#222222
    classDef decision fill:#f7c6c7,stroke:#35424a,stroke-width:3px,color:#222222
    classDef subgraphLabel font-size:18px, stroke-width:5px, font-weight:bold
    class Data_Preprocessing,Model_Training,XAI_Integration,Performance_Evaluation subgraphLabel
    
    linkStyle 0 stroke:#ff6b6b,stroke-width:2px,fill:none
    linkStyle 1 stroke:#ff6b6b,stroke-width:2px,fill:none
    linkStyle 2 stroke:#ff6b6b,stroke-width:2px,fill:none
    linkStyle 3 stroke:#ff6b6b,stroke-width:2px,fill:none
    linkStyle 4 stroke:#ff6b6b,stroke-width:2px,fill:none
    linkStyle 5 stroke:#77DD77,stroke-width:2px,fill:none
    linkStyle 6 stroke:#77DD77,stroke-width:2px,fill:none
    linkStyle 7 stroke:#77DD77,stroke-width:2px,fill:none
    linkStyle 8 stroke:#77DD77,stroke-width:2px,fill:none
    linkStyle 9 stroke:#77DD77,stroke-width:2px,fill:none
    linkStyle 10 stroke:#77DD77,stroke-width:2px,fill:none
    linkStyle 11 stroke:#77DD77,stroke-width:2px,fill:none
    linkStyle 12 stroke:#77DD77,stroke-width:2px,fill:none
    linkStyle 13 stroke:#77DD77,stroke-width:2px,fill:none
    linkStyle 14 stroke:#77DD77,stroke-width:2px,fill:none
    linkStyle 15 stroke:#d277ff,stroke-width:2px,fill:none
    linkStyle 16 stroke:#d277ff,stroke-width:2px,fill:none
    linkStyle 17 stroke:#d277ff,stroke-width:2px,fill:none
    linkStyle 18 stroke:#d277ff,stroke-width:2px,fill:none
    linkStyle 19 stroke:#d277ff,stroke-width:2px,fill:none
    linkStyle 20 stroke:#d277ff,stroke-width:2px,fill:none
    linkStyle 21 stroke:#d277ff,stroke-width:2px,fill:none
    linkStyle 22 stroke:#d277ff,stroke-width:2px,fill:none
    linkStyle 23 stroke:#d277ff,stroke-width:2px,fill:none
    linkStyle 24 stroke:#ffcc00,stroke-width:2px,fill:none
    linkStyle 25 stroke:#ffcc00,stroke-width:2px,fill:none
    linkStyle 26 stroke:#ffcc00,stroke-width:2px,fill:none
    linkStyle 27 stroke:#ffcc00,stroke-width:2px,fill:none
    linkStyle 28 stroke:#d277ff,stroke-width:2px,fill:none
    linkStyle 29 stroke:#d277ff,stroke-width:2px,fill:none
    linkStyle 30 stroke:#d277ff,stroke-width:2px,fill:none
    linkStyle 31 stroke:#ff9797,stroke-width:3px,fill:none
    linkStyle 32 stroke:#ff9797,stroke-width:3px,fill:none
    linkStyle 33 stroke:#d277ff,stroke-width:2px,fill:none
    linkStyle 34 stroke:#d277ff,stroke-width:2px,fill:none
    linkStyle 35 stroke:#d277ff,stroke-width:2px,fill:none
```

<p align="center"><em>System Architecture Diagram</em></p>

## Data Sources

- **Synthetic Data Generation**: To generate synthetic credit card transaction data, including fraudulent transactions, refer to the [Sparkov Data Generation repository](https://github.com/namebrandon/Sparkov_Data_Generation).
- **Combined Dataset**: The combined dataset from Sparkov Data Generation, converted into a standard format, can be accessed [here on Kaggle](https://www.kaggle.com/datasets/kartik2112/fraud-detection).

This repository is a valuable resource for researchers and practitioners interested in the intersection of fraud detection and explainable AI, providing a foundation for further exploration and development in this critical area of financial security.

This repository provides a comprehensive resource for researchers and practitioners focused on leveraging explainable AI to improve the transparency and performance of deep learning models in credit card fraud detection.