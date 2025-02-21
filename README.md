# Credit-Card-Transaction-Fraud-Detection-Using-Explainable-AI

### Description
This project contains a credit card fraud detection system that leverages advanced machine learning algorithms, including neural networks and random forests, to identify fraudulent transactions. The project utilizes synthetic data from the Sparkov dataset of the credit card fraud detection dataset to train and evaluate the model.

Key features of the project include:
- **Model Development**: Custom-built fraud detection models using various machine learning techniques, ensuring adaptability and performance.
- **Explainable AI Integration**: 
  - SHAP (SHapley Additive exPlanations) implementation with interactive visualizations:
    - Force plots showing feature contributions for individual predictions
    - Summary plots highlighting global feature importance
    - Dependence plots showing how specific features impact predictions
  - LIME (Local Interpretable Model-agnostic Explanations) for local prediction explanations
- **Testing and Evaluation**: Rigorous testing on both synthetic datasets, with performance metrics and  against the ground truth to make sure the explainability methods are working as expected.
- **Visualizations**: The project includes visualizations that allow users to:
  - Explore feature contributions in detail
  - Understand model decisions at both global and local levels
  - Analyze feature dependencies and their impact on predictions
- **Resources**: The project utilizes popular machine learning libraries:
  - TensorFlow and Scikit-learn for model development
  - SHAP and LIME for explainability
  - Faker for synthetic data generation

### Data Sources
- **Synthetic Data Generation**: To generate synthetic credit card transaction data, including fraudulent transactions, refer to the [Sparkov Data Generation repository](https://github.com/namebrandon/Sparkov_Data_Generation).
- **Combined Dataset**: The combined dataset from Sparkov Data Generation, converted into a standard format, can be accessed [here on Kaggle](https://www.kaggle.com/datasets/kartik2112/fraud-detection).

This repository is a valuable resource for researchers and practitioners interested in the intersection of fraud detection and explainable AI, providing a foundation for further exploration and development in this critical area of financial security.

### Links & References:
[Jira Board](https://laiminhthong1.atlassian.net/jira/core/projects/CCTFDUX/board?atlOrigin=eyJpIjoiMGVjZTM2MzYxZGEyNGY3Y2E1ZGU1ODIzYjdkMTU0MzgiLCJwIjoiaiJ9)