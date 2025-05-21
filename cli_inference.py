#!/usr/bin/env python3
import sys
from inference import build_model, run_ve, run_bp, run_lw

def prompt_choice(prompt, options):
    while True:
        print(prompt)
        for i, opt in enumerate(options, 1):
            print(f"{i}. {opt}")
        choice = input("Enter number: ")
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1]
        print("Invalid choice, try again.\n")

def prompt_evidence():
    evidence = {}
    for var, options in [
        ("Income_Level", ["Low", "Medium", "High"]),
        ("Experience_Level", ["Junior", "Mid", "Senior"]),
        ("House_Ownership", ["Yes", "No", "Unknown"]),
        ("Car_Ownership", ["Yes", "No", "Unknown"])
    ]:
        val = prompt_choice(f"Select {var}:", options)
        evidence[var] = val
    return evidence

def main():
    model = build_model()

    print("Credit Risk Bayesian Network Inference")
    algorithm = prompt_choice("Choose inference algorithm:", ["ve", "bp", "lw"])

    evidence = prompt_evidence()

    print("\nRunning inference...\n")
    if algorithm == "ve":
        result = run_ve(model, evidence)
        for state, prob in zip(result.state_names["Risk_Flag"], result.values):
            print(f"P(Risk_Flag={state}) = {prob:.4f}")
    elif algorithm == "bp":
        result = run_bp(model, evidence)
        for state, prob in zip(result.state_names["Risk_Flag"], result.values):
            print(f"P(Risk_Flag={state}) = {prob:.4f}")
    else:
        probs, ci = run_lw(model, evidence)
        for state in probs:
            lo, hi = ci[state]
            print(f"P(Risk_Flag={state}) ≈ {probs[state]:.4f} (95% CI: [{lo:.4f}, {hi:.4f}])")

if __name__ == "__main__":
    main()
