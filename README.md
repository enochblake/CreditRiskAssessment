# Credit Risk Assessment Using Bayesian Networks

## Overview
This project leverages Bayesian Networks to model and infer credit risk based on customer attributes such as income level, experience, home ownership, and car ownership. The solution includes data preprocessing, model training, and inference capabilities to predict "HighRisk" or "LowRisk" classifications.

## Key Features
- **Data Preprocessing**: Discretizes continuous features and maps binary flags.
- **Model Training**: Builds a Bayesian Network and learns Conditional Probability Tables (CPTs) using Maximum Likelihood Estimation.
- **Inference**: Supports Variable Elimination, Belief Propagation, and Likelihood Weighting for single and batch predictions.

## Installation

1. **Setup Python environment**  
   Ensure you have Python 3.8+ installed. It is recommended to use a virtual environment or Anaconda.

2. **Install required packages**  
   Run the following commands to install dependencies:
   ```bash
   pip install pgmpy pandas numpy matplotlib networkx flask argparse

## Repository Structure
CreditRiskAssessment/
├── data/
│   └── credit_risk_dataset.csv     # Raw credit risk data
├── learn.py                        # Script to preprocess data and train the model
├── inference.py                    # Script to perform inference
├── plot_bn.py                     # Script to visualize Bayesian Network graph
├── generate_samples.py             # Script to generate synthetic samples using the model
├── flask_app.py                    # Minimal Flask web app for interactive inference
├── cpds.pkl                       # Generated CPTs after running learn.py
└── README.md                      # Project documentation

## Usage
**Train the Model**
python learn.py
This will load data, preprocess it, learn the CPDs, and save them in cpds.pkl.
**Visualize the Bayesian Network Graph**
python plot_bn.py
This opens a window displaying the network structure.
**Generate Synthetic Samples**
python generate_samples.py
Prints synthetic data samples with predicted Risk_Flag values.
**Perform Inference**
Single Inference via Command Line
Specify all evidence variables and algorithm:
python inference.py \
  --algorithm ve \
  --income-level Medium \
  --experience-level Mid \
  --house-ownership Yes \
  --car-ownership No

*Supported algorithms:*
ve — Variable Elimination (exact inference)
bp — Belief Propagation (approximate)
lw — Likelihood Weighting (sampling-based with confidence intervals)

*Batch Inference with CSV Input*
Prepare a CSV file batch.csv with columns:
Income_Level,Experience_Level,House_Ownership,Car_Ownership
Medium,Mid,Yes,No
High,Senior,No,Unknown

*Run batch inference:*
python inference.py --algorithm ve --batch-file batch.csv
The output will be a CSV printed to the console with probability results.

**Run the Flask Web App for Interactive Inference**
Install Flask:
pip install Flask
Run the app:
python flask_app.py

Open your browser at http://127.0.0.1:5000/ and use the form to input evidence and select inference algorithms.

## Notes
Ensure cpds.pkl exists by running learn.py before inference.

The inference script requires all evidence variables for single inference mode.

Visualization depends on networkx and matplotlib.

## License
MIT License

## References
pgmpy Documentation: https://pgmpy.org
Koller, D., & Friedman, N. (2009). Probabilistic Graphical Models.
Dataset adapted from Kaggle’s Credit Risk Dataset.