import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# make sure relative imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.dgp import generate_synthetic_data
from estimators.dr_estimator import dr_estimator


def monte_carlo(num_trials=50, num_samples=1000, num_features=4, num_splits=4, dr_seed=42):
    true_atts = []
    estimated_atts = []

    for seed in range(num_trials):
        print(f"Running trial {seed + 1}/{num_trials} with seed {seed}...")
        data = generate_synthetic_data(
            num_samples=num_samples,
            num_features=num_features,
            random_seed=seed
        )

        est_att = dr_estimator(data, n_splits=num_splits, seed=dr_seed)
        # Calculate the true ATT
        estimated_atts.append(est_att)

    return np.array(estimated_atts)


def plot_results(true_atts, estimated_atts):
    df = pd.DataFrame({
        'True ATT': true_atts,
        'Estimated ATT': estimated_atts
    })

    plt.figure(figsize=(8, 6))
    df.boxplot(column=['Estimated ATT'])
    plt.axhline(y=np.mean(true_atts), color='r', linestyle='--', label='Mean True ATT')
    plt.title("Outcome Regression Estimator")
    plt.ylabel("ATT")
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    # Set parameters
    num_trials = 5
    seed = 42
    num_samples = 1000
    num_features = 4
    num_splits = 5
    estimated_atts = monte_carlo(num_trials=num_trials, num_samples=num_samples, num_features=num_features,
                                 num_splits=num_splits, dr_seed=seed)

    print(f"Estimated ATTs: {estimated_atts}")

    # plot_results(true_atts, estimated_atts)


if __name__ == "__main__":
    main()
