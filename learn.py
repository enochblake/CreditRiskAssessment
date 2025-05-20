#!/usr/bin/env python3
"""
learn.py

– Loads raw credit data from data/credit_risk_dataset.csv
– Preprocesses (discretizes continuous features, maps binary flags)
– Defines a BayesianModel structure
– Learns CPTs via Maximum Likelihood Estimation
– Validates normalization, pickles CPDs to cpds.pkl
"""

import os
import pickle
import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator

RAW_CSV = os.path.join("data", "credit_risk_dataset.csv")
PICKLE_PATH = "cpds.pkl"

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # Discretize Income and Experience
    df["Income_Level"] = pd.cut(
        df["Income"],
        bins=[-1, 50000, 100000, df["Income"].max()],
        labels=["Low", "Medium", "High"]
    )
    df["Experience_Level"] = pd.cut(
        df["Experience"],
        bins=[-1, 2, 5, df["Experience"].max()],
        labels=["Junior", "Mid", "Senior"]
    )

    # Map binary flags to strings
    df["House_Ownership"] = df["House_Ownership"].map({0: "No", 1: "Yes"})
    df["Car_Ownership"]   = df["Car_Ownership"].map({0: "No", 1: "Yes"})

    # Map target to categorical
    df["Risk_Flag"] = df["Risk_Flag"].map({0: "LowRisk", 1: "HighRisk"})

    # Keep only the columns we model
    cols = [
        "Income_Level",
        "Experience_Level",
        "House_Ownership",
        "Car_Ownership",
        "Risk_Flag",
    ]
    return df[cols].dropna()

def build_and_learn(df: pd.DataFrame) -> BayesianModel:
    # Define network structure
    structure = [
        ("Income_Level",    "Risk_Flag"),
        ("Experience_Level","Risk_Flag"),
        ("House_Ownership", "Risk_Flag"),
        ("Car_Ownership",   "Risk_Flag"),
    ]
    model = BayesianModel(structure)
    model.fit(df, estimator=MaximumLikelihoodEstimator)
    return model

def validate_cpds(model: BayesianModel):
    for cpd in model.get_cpds():
        values = cpd.get_values().reshape(cpd.cardinality, -1)
        for idx, row in enumerate(values):
            if not abs(row.sum() - 1.0) < 1e-6:
                raise ValueError(
                    f"Normalization error in CPT for {cpd.variable}, row {idx}"
                )

def main():
    if not os.path.exists(RAW_CSV):
        raise FileNotFoundError(f"Raw data file not found: {RAW_CSV}")

    print(f"[learn.py] Loading raw data from {RAW_CSV}")
    raw_df = pd.read_csv(RAW_CSV)

    print("[learn.py] Preprocessing data...")
    data = preprocess(raw_df)

    print("[learn.py] Learning Bayesian Network CPDs...")
    model = build_and_learn(data)

    print("[learn.py] Validating learned CPDs...")
    validate_cpds(model)

    print(f"[learn.py] Pickling CPDs to {PICKLE_PATH}")
    with open(PICKLE_PATH, "wb") as f:
        pickle.dump(model.get_cpds(), f)

    print("[learn.py] All CPDs learned and validated successfully.")

if __name__ == "__main__":
    main()
