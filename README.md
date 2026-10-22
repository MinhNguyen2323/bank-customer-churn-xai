# Bank Customer Churn Prediction with Explainable AI

This repository contains the reproducibility code and decision-support prototype developed for a graduation thesis on bank customer churn prediction.

The study compares classical machine-learning and deep tabular models under a leakage-controlled experimental framework and evaluates predictive performance, calibration, statistical uncertainty, explainability, robustness, subgroup behavior, and computational efficiency.

## Study Overview

Nine predictive models were evaluated:

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- CatBoost
- Multilayer Perceptron
- TabNet
- FT-Transformer
- RealTabR

The experiments use the public **Churn for Bank Customers** dataset containing 10,000 customer records.

The final benchmark identified XGBoost as a strong performance-efficiency compromise for this dataset, with:

- ROC-AUC: **0.8707**
- PR-AUC: **0.7223**
- Brier Score: **0.0986**

CatBoost obtained the highest ROC-AUC at **0.8711**.

The study does not claim that XGBoost is universally superior or that the resulting prototype is production-ready.

## Repository Structure

```text
research/
    bankpaper_clean_reproducible.ipynb

application/
    app.py
    model_utils.py
    train_model.py
    study_evidence.json
    artifacts/

data/
    churn.csv

requirements.txt
README.md
```

## Reproducibility Notebook

`research/bankpaper_clean_reproducible.ipynb` contains a consolidated implementation of the experimental workflow.

The pipeline includes:

1. Data verification
2. Leakage-controlled development/holdout splitting
3. Hyperparameter optimization
4. Repeated cross-validation
5. Ablation studies
6. Probability calibration and threshold selection
7. Final holdout evaluation
8. Statistical inference
9. SHAP explainability
10. Robustness and subgroup analyses
11. Computational assessment

The notebook is organized as a clean sequential implementation for reproducibility. The numerical results reported in the thesis originate from the archived original experimental process.

## Decision-Support Application

The Streamlit application operationalizes the selected XGBoost model as a research prototype.

It provides four functions:

- **Model Overview**: benchmark performance and global model information
- **Individual Prediction**: customer-level churn probability and SHAP explanation
- **Portfolio Prioritization**: ranking customers according to predicted churn risk
- **Transparency & Governance**: robustness, subgroup, and model-sensitivity evidence

The application is intended for demonstration and decision-support research only. It does not provide automated retention decisions or causal recommendations.

## Running the Application

Create a virtual environment and install the dependencies:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install packages:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run application/app.py
```

## Rebuilding the Demo Model

The repository already contains the model artifacts required to run the application.

To rebuild the application model from the dataset:

```bash
python application/train_model.py
```

The XGBoost application model follows the experimental configuration used in the study.

## Dataset

Dataset:

**Churn for Bank Customers**

Source: Kaggle  
https://www.kaggle.com/datasets/mathchi/churn-for-bank-customers

The predictive variables are:

- CreditScore
- Geography
- Gender
- Age
- Tenure
- Balance
- NumOfProducts
- HasCrCard
- IsActiveMember
- EstimatedSalary

The target variable is `Exited`.

## Important Note

This repository distinguishes between:

1. the **original research experiments**, from which the thesis results and limitations were reported; and
2. the **consolidated reproducibility implementation**, which reorganizes the final experimental methodology into a clean sequential pipeline.

The consolidated implementation does not retroactively replace the archived experimental results reported in the thesis.

## Disclaimer

This project is an academic research prototype.

The dataset comes from a single public benchmark and does not provide independent institutional or temporal validation. The application should therefore not be interpreted as a production-ready banking decision system.
