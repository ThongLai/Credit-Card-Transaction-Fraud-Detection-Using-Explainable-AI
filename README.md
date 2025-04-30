# Explainable AI (XAI) in Deep Learning Models for Credit Card Fraud Detection

[![Python](https://img.shields.io/badge/python-3.10.11-blue.svg)](https://www.python.org/downloads/release/python-31011/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10.1-orange.svg)](https://www.tensorflow.org/versions/r2.10/api_docs)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

- **Documentation:** [Report Documentation.pdf
](https://docs.google.com/viewer?url=github.com/ThongLai/Credit-Card-Transaction-Fraud-Detection-Using-Explainable-AI/blob/main/Report%20Documentation.pdf?raw=true)

- **Main analysis notebook:** [XAI_methods.ipynb](https://github.com/ThongLai/Credit-Card-Transaction-Fraud-Detection-Using-Explainable-AI/blob/main/XAI_methods.ipynb)

- **Run the live main notebook:** [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ThongLai/Credit-Card-Transaction-Fraud-Detection-Using-Explainable-AI/main?urlpath=%2Fdoc%2Ftree%2FXAI_methods.ipynb)

- **Models:** [architectures](https://github.com/ThongLai/Credit-Card-Transaction-Fraud-Detection-Using-Explainable-AI/tree/main/architectures)<a name="models" id="models"></a>

## Description

Credit card fraud represents an important and growing challenge in the financial sector, causing significant monetary losses worldwide. According to the data from the UK Finance, fraudsters stole over £1.3 billion in 2021 alone through authorised and unauthorised fraud, with card fraud accounting for a significant part of these damages [^ukfinance2022]. Detection of fraud transactions presents several important challenges that increase the need for advanced computational approaches.

The project utilized and evaluated the **XAI methods** on **Deep Learning** architectures that are widely known to be used in detecting credit card transaction fraud, including **CNN** and **LSTM** with attention mechanisms, trained on **Sparkov's** synthetic dataset. The main contribution lies in the integration and comparative analysis of three Explainable AI methods: **SHAP**, **LIME** and **Anchors**. Research further evaluates the effectiveness of each XAI method based on **Faithfulness**, **Monotonicity**, and **Completeness** metrics.

<p align="center">
  <a href="https://docs.google.com/viewer?url=github.com/ThongLai/Credit-Card-Transaction-Fraud-Detection-Using-Explainable-AI/blob/main/poster.pdf?raw=true">
    <img src="visualisation/poster.jpg" alt="Academic Poster" width="600">
  </a>
  <br>
  <em>Academic Poster</em>
  <br>
  <em><small><a href="https://docs.google.com/viewer?url=github.com/ThongLai/Credit-Card-Transaction-Fraud-Detection-Using-Explainable-AI/blob/main/poster.pdf?raw=true">(Click to view PDF)</a></small></em>
</p>

**Key Features:**
- **Multi-Model Comparison:**  
  Assess and compare different deep learning architectures for fraud detection.
- **Explainable AI Integration:**  
  - **SHAP:** Interactive visualisations (force plots, summary plots, dependence plots) that elucidate feature contributions.  
  - **LIME:** Local interpretable explanations (in progress) to detail individual predictions.  
  - **Anchors:** Rule-based explanations offering high-precision insights.
- **Robust Evaluation:**  
  - Visual tools to explore feature impacts locally, supporting transparent decision-making.
  - Extensive testing on synthetic data with XAI performance metrics to validate both model interpretability.

## System Architecture

The fraud detection system using XAI comprises several integrated modules:

1. **Data Collection & Model Design:**  
  Gathers synthetic transaction data (**Sparkov** dataset) and design model architectures (CNN, LSTM). Data is loaded and versioned for reproducibility.
2. **Data Preprocessing:**  
  Raw transaction data is cleaned, normalised, and balanced using the **SMOTE** method.
3. **Model Training:**  
  Separate pipelines are implemented for training the **CNN** and **LSTM** models.
4. **XAI Integration:**  
  Post-training, XAI techniques are applied to generate explanations
for model predictions.
5. **Explainability Performance Evaluation:**  
  Comprehensive XAI evaluation metrics, including **Faithfulness**, **Monotonicity**, and **Completeness** are computed.

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
<p align="center">
  <br>
  <em>System Architecture Diagram</em>
  <br>
  <em><small>Can't see the diagram? <a href="https://github.com/ThongLai/Credit-Card-Transaction-Fraud-Detection-Using-Explainable-AI/blob/main/visualisation/system_architecture.png">View system architecture image</a></small></em>
</p>


## Data Sources

- **Synthetic Data Generation**: To generate synthetic credit card transaction data, including fraudulent transactions, refer to the [Sparkov Data Generation repository](https://github.com/namebrandon/Sparkov_Data_Generation).
- **Combined Dataset**: The combined dataset from Sparkov Data Generation, converted into a standard format, can be accessed [here on Kaggle](https://www.kaggle.com/datasets/kartik2112/fraud-detection).

## XAI Methods Evaluation

### Summary Performance Results
| Method  | Faithfulness | Monotonicity | Completeness |
|---------|--------------|--------------|--------------|
| SHAP    | 0.602        | 0.447        | 0.171        |
| LIME    | 0.325        | 0.467        | 0.217        |
| Anchors | 0.364        | 0.478        | 0.028        |

## Key Visualizations

### Feature Importance Analysis
<p align="center">
  <img src="https://github.com/ThongLai/Credit-Card-Transaction-Fraud-Detection-Using-Explainable-AI/blob/main/visualisation/shap_summary_global_of_CNN.png?raw=true" alt="SHAP Feature Importance" width="500">
  <br>
  <em>SHAP values showing global feature importance across the dataset</em>
</p>

### Local Explanation Example
<p align="center">
  <img src="https://github.com/ThongLai/Credit-Card-Transaction-Fraud-Detection-Using-Explainable-AI/blob/main/visualisation/lime_explain_plot_of_LSTM_at_1044.png?raw=true" alt="LIME Explanation" width="500">
  <br>
  <em>LIME explaining an individual fraud prediction</em>
</p>

### Rule-Based Insights
<p align="center">
  <img src="https://github.com/ThongLai/Credit-Card-Transaction-Fraud-Detection-Using-Explainable-AI/blob/main/visualisation/anchors_notebook_of_LSTM_at_2025.png?raw=true" alt="Anchors Rules" width="1000">
  <br>
  <em>Anchors generating interpretable rules for fraud detection</em>
</p>

## Key Findings
- **Explainability Analysis**:
  - **SHAP** demonstrated the highest faithfulness (0.602), providing the most reliable feature attributions.
  - **LIME** showed balanced performance across metrics with good completeness (0.217).
  - **Anchors** excelled in monotonicity (0.478) but with limited coverage (0.028).
- **Feature Importance**: Transaction `amount`, `merchant` `category`, and transaction `hou`r emerged as the most influential features across both models.
- **Confidence Analysis**: XAI methods showed varying performance across prediction confidence levels, with SHAP maintaining the most consistent performance (0.544-0.629 faithfulness) across all confidence bins.

This repository is a valuable resource for researchers and practitioners interested in the intersection of fraud detection and explainable AI , providing a foundation for further exploration and development in this critical area of financial security.

## Project Directory Hierarchy
```
Project Root/
├── XAI_methods.ipynb ............................ Explainable AI methods implementation  
├── Siddhartha_CNN.ipynb ......................... CNN model implementation  
├── Ibtissam_LSTM.ipynb .......................... LSTM model implementation  
├── utils.py ..................................... Utility functions  
├── requirements.txt ............................. Dependencies  

├── architectures/ ............................... Trained model storage  
│   ├── model_1_Siddhartha_CNN_acc99/  
│   └── model_2_Ibtissam_LSTM_acc98/  
├── data/ ........................................ Data files and results  
│   ├── predictions.csv  
│   ├── stratified_samples.csv  
│   └── xai_metrics.json  
├── visualisation/ ............................... Visualisation outputs  
├── README.md .................................... Project brief  
├── Report Documentation.pdf ..................... Project Report Documentation
├── presentation.pdf ............................. Project Presentation
├── poster.pdf ................................... Academic Poster
└── .gitignore ................................... Git configuration
```

[^ukfinance2022]: UK Finance. (2022). *Annual Fraud Report 2022*. https://www.ukfinance.org.uk/policy-and-guidance/reports-and-publications/annual-fraud-report-2022