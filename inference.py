#!/usr/bin/env python3
"""
inference.py

– Loads the network structure and pickled CPDs from cpds.pkl
– Accepts command-line evidence for single inference or batch CSV
– Supports:
    • Variable Elimination (ve)
    • Belief Propagation (bp)
    • Likelihood Weighting sampling (lw) with 95% CIs
"""

import argparse
import pickle
import os
import pandas as pd
import numpy as np
from pgmpy.models import BayesianModel
from pgmpy.inference import VariableElimination, BeliefPropagation, LikelihoodWeighting

PICKLE_PATH = "cpds.pkl"

def build_model() -> BayesianModel:
    # Define network structure (must match learn.py)
    structure = [
        ("Income_Level",    "Risk_Flag"),
        ("Experience_Level","Risk_Flag"),
        ("House_Ownership", "Risk_Flag"),
        ("Car_Ownership",   "Risk_Flag"),
    ]
    model = BayesianModel(structure)

    if not os.path.exists(PICKLE_PATH):
        raise FileNotFoundError(
            f"CPD pickle not found: {PICKLE_PATH}\n"
            "Run `python learn.py` first to generate it."
        )

    with open(PICKLE_PATH, "rb") as f:
        cpds = pickle.load(f)

    model.add_cpds(*cpds)
    if not model.check_model():
        raise ValueError("Model check failed; CPDs may not match structure.")
    return model

def run_ve(model, evidence):
    return VariableElimination(model).query(
        variables=["Risk_Flag"], evidence=evidence, show_progress=False
    )

def run_bp(model, evidence):
    return BeliefPropagation(model).query(
        variables=["Risk_Flag"], evidence=evidence, show_progress=False
    )

def run_lw(model, evidence, n_samples=5000):
    sampler = LikelihoodWeighting(model)
    samples = sampler.run(evidence=evidence, variables=["Risk_Flag"], n_samples=n_samples)
    counts = samples.state_counts["Risk_Flag"]
    total = counts.sum()
    probs = {state: counts[state] / total for state in counts.index}
    # 95% confidence intervals
    ci = {}
    z = 1.96
    for state, p in probs.items():
        se = np.sqrt(p * (1 - p) / total)
        ci[state] = (max(0, p - z * se), min(1, p + z * se))
    return probs, ci

def infer_single(args, model):
    evidence = {
        "Income_Level":     args.income_level,
        "Experience_Level": args.experience_level,
        "House_Ownership":  args.house_ownership,
        "Car_Ownership":    args.car_ownership,
    }

    print(f"[inference.py] Performing single inference using '{args.algorithm}'")
    if args.algorithm == "ve":
        result = run_ve(model, evidence)
        for state, prob in zip(result.state_names["Risk_Flag"], result.values):
            print(f"P(Risk_Flag={state}) = {prob:.4f}")
    elif args.algorithm == "bp":
        result = run_bp(model, evidence)
        for state, prob in zip(result.state_names["Risk_Flag"], result.values):
            print(f"P(Risk_Flag={state}) = {prob:.4f}")
    else:  # likelihood weighting
        probs, ci = run_lw(model, evidence, n_samples=args.samples)
        for state in probs:
            lo, hi = ci[state]
            print(f"P(Risk_Flag={state}) ≈ {probs[state]:.4f} (95% CI: [{lo:.4f}, {hi:.4f}])")

def infer_batch(args, model):
    if not os.path.exists(args.batch_file):
        raise FileNotFoundError(f"Batch file not found: {args.batch_file}")

    df = pd.read_csv(args.batch_file)
    results = []

    for _, row in df.iterrows():
        evidence = row.to_dict()
        if args.algorithm == "lw":
            probs, ci = run_lw(model, evidence, n_samples=args.samples)
            entry = {**evidence}
            for state in probs:
                entry[f"P_{state}"] = probs[state]
                entry[f"CI_{state}"] = ci[state]
        else:
            result = run_ve(model, evidence) if args.algorithm == "ve" else run_bp(model, evidence)
            entry = {**evidence}
            for state, prob in zip(result.state_names["Risk_Flag"], result.values):
                entry[f"P_{state}"] = prob
        results.append(entry)

    out_df = pd.DataFrame(results)
    print(out_df.to_csv(index=False))

def parse_args():
    p = argparse.ArgumentParser(description="Inference on CreditRisk BN")
    p.add_argument(
        "--algorithm",
        choices=["ve", "bp", "lw"],
        default="ve",
        help="ve=VariableElimination, bp=BeliefPropagation, lw=LikelihoodWeighting"
    )
    p.add_argument("--income-level",
                   choices=["Low", "Medium", "High"],
                   help="Discretized income level")
    p.add_argument("--experience-level",
                   choices=["Junior", "Mid", "Senior"],
                   help="Discretized experience level")
    p.add_argument("--house-ownership",
                   choices=["Yes", "No"],
                   help="House ownership status")
    p.add_argument("--car-ownership",
                   choices=["Yes", "No"],
                   help="Car ownership status")
    p.add_argument("--batch-file",
                   type=str,
                   help="Path to CSV with columns: Income_Level,Experience_Level,House_Ownership,Car_Ownership")
    p.add_argument("--samples",
                   type=int,
                   default=5000,
                   help="Number of samples for likelihood weighting (lw)")
    return p.parse_args()

def main():
    args = parse_args()
    model = build_model()

    if args.batch_file:
        infer_batch(args, model)
    else:
        required = ["income_level", "experience_level", "house_ownership", "car_ownership"]
        missing = [r for r in required if getattr(args, r) is None]
        if missing:
            raise ValueError(f"Missing required arguments for single inference: {missing}")
        infer_single(args, model)

if __name__ == "__main__":
    main()
