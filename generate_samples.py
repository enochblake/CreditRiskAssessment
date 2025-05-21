from inference import build_model  # Again, ensure build_model exists here
import pandas as pd

def generate_samples(n=10):
    model = build_model()
    samples = model.simulate(n)
    print("Generated synthetic samples with predicted Risk_Flag:\n")
    print(samples)

if __name__ == "__main__":
    generate_samples()
