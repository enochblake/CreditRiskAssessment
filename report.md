# Credit Risk Assessment Using Bayesian Networks: Technical Report

## 1. Introduction
This project addresses credit risk prediction using Bayesian Networks. The goal is to infer the probability of a customer defaulting (Risk_Flag) based on financial and demographic attributes.

## 2. Domain and Problem Statement
**Domain**: Financial Services (Credit Risk Prediction).  
**Problem**: Traditional credit scoring methods lack transparency in probabilistic reasoning. Bayesian Networks provide an interpretable framework to model dependencies between risk factors.

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

![Bayesian Network Diagram](image.png)  
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

## 6.Implementation
**Tools**:
pgmpy for probabilistic graphical modeling, pandas and numpy for data handling.
**Scripts**:
learn.py: Loads raw data, preprocesses, and trains the Bayesian Network.

inference.py: Performs inference with multiple algorithms.

plot_bn.py: Visualizes the network structure.

generate_samples.py: Produces synthetic samples from the learned model.

Flask web app for interactive inference.
 
## 7.Validation
Model CPDs are checked for probabilistic consistency.

Synthetic samples are generated to verify plausible predictions.

## 8.Limitations and Future Work
Simplified network structure may omit complex dependencies.

No quantitative accuracy metrics such as ROC-AUC currently.

Future plans include expanding features, integrating real-time inference, and validating with real-world credit data.

## 8.Summary
This project demonstrates the application of Bayesian Networks for credit risk assessment, offering a probabilistic and interpretable approach to predict default risk. The model integrates key financial and demographic factors, learns their relationships from data, and supports multiple inference algorithms for flexible and robust prediction. While the current implementation provides a solid foundation, future enhancements could include more complex features and real-world validation to improve accuracy and applicability in credit decision-making processes.

## 9.References
pgmpy Documentation: https://pgmpy.org

Koller, D., & Friedman, N. (2009). Probabilistic Graphical Models.

Dataset adapted from Kaggle’s Credit Risk Dataset