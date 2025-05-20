# CreditRiskAssessment# Credit Risk Assessment Using Bayesian Networks

## Overview
This project leverages Bayesian Networks to model and infer credit risk based on customer attributes such as income level, experience, home ownership, and car ownership. The solution includes data preprocessing, model training, and inference capabilities to predict "HighRisk" or "LowRisk" classifications.

## Key Features
- **Data Preprocessing**: Discretizes continuous features and maps binary flags.
- **Model Training**: Builds a Bayesian Network and learns Conditional Probability Tables (CPTs) using Maximum Likelihood Estimation.
- **Inference**: Supports Variable Elimination, Belief Propagation, and Likelihood Weighting for single and batch predictions.

## Installation
1. **Dependencies**:  
   Ensure Python 3.8+ is installed. Install required packages:  
   ```bash
   pip install pgmpy pandas numpy argparse

   Repository Structure:
   ├── data/credit_risk_dataset.csv   # Raw credit risk data
├── learn.py                       # Script to preprocess data and train the model
├── inference.py                   # Script to perform inference
├── cpds.pkl                       # Generated CPTs after running learn.py
└── README.md                      # Project documentation

Usage:

Train the Model:
bash
python learn.py
Single Inference:
python inference.py \
  --algorithm ve \
  --income-level Medium \
  --experience-level Mid \
  --house-ownership Yes \
  --car-ownership No

  Batch Inference:
  Save input data as batch.csv with columns: Income_Level,Experience_Level,House_Ownership,Car_Ownership. Run:
  python inference.py --algorithm ve --batch-file batch.csv

License
© 2025 [Organization Name]. Proprietary and confidential.

---

**report.md**  
```markdown
# Credit Risk Assessment Using Bayesian Networks: Technical Report

## 1. Introduction
This project addresses credit risk prediction using Bayesian Networks. The goal is to infer the probability of a customer defaulting (Risk_Flag) based on financial and demographic attributes.

## 2. Domain and Problem Statement
**Domain**: Financial Services (Credit Risk Prediction).  
**Problem**: Traditional credit scoring methods lack transparency in probabilistic reasoning. Bayesian Networks provide a interpretable framework to model dependencies between risk factors.

## 3. Data and Preprocessing
- **Dataset**: `credit_risk_dataset.csv` containing customer profiles and loan details.
- **Preprocessing**:  
  - Discretized `Income` into Low (<50k), Medium (50k–100k), and High (>100k).  
  - Discretized `Experience` into Junior (<2 years), Mid (2–5 years), and Senior (>5 years).  
  - Mapped binary flags (`House_Ownership`, `Car_Ownership`) to "Yes"/"No".  

## 4. Bayesian Network Design
### Structure
- **Nodes**: `Income_Level`, `Experience_Level`, `House_Ownership`, `Car_Ownership` (parents) → `Risk_Flag` (child).  
- **Edges**: Direct dependencies from attributes to Risk_Flag.  

![Bayesian Network Diagram](diagram.png)  
*Diagram: All parent nodes influence the target variable Risk_Flag.*

### Conditional Probability Tables (CPTs)
- **Priors**: Learned from data using Maximum Likelihood Estimation.  
- **Posteriors**: Validated for normalization (sum to 1 ± 1e-6).  

## 5. Inference
**Algorithms**:  
1. **Variable Elimination (VE)**: Exact inference for precise probabilities.  
2. **Belief Propagation (BP)**: Approximate inference for faster results.  
3. **Likelihood Weighting (LW)**: Sampling-based method with 95% confidence intervals.  

**Example Query**:  
```python
P(Risk_Flag | Income_Level=Medium, Experience_Level=Mid, House_Ownership=Yes, Car_Ownership=No)

Implementation
Tools: pgmpy for modeling, pandas for data handling.

Validation: CPTs checked for probabilistic consistency.

Limitations:

Simplified network structure; domain knowledge could refine edges.

No quantitative accuracy metrics (e.g., ROC-AUC).

Future Work
Integrate with real-time credit approval systems.

Expand features (e.g., loan history, credit scores).

Validate against ground-truth defaults.

References
pgmpy Documentation: https://pgmpy.org

Koller, D., & Friedman, N. (2009). Probabilistic Graphical Models.

Dataset adapted from Kaggle’s Credit Risk Dataset.