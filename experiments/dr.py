import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# make sure relative imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.dgp import generate_synthetic_data
from estimators.or_estimator import or_estimator
from estimators.ipw_estimator import ipw_estimator


def compute_true_att(data):
    treated = data.loc[(data['G'] == 1) & (data['D'] == 1)].copy()
    return np.mean(treated['Y_1_1'].values) - np.mean(treated['Y_1_0'].values)


def monte_carlo(num_trials=50, num_samples=3000, num_features=3):
    true_atts = []
    estimated_atts = []

    for seed in range(num_trials):
        print(f"Running trial {seed + 1}/{num_trials} with seed {seed}...")
        data = generate_synthetic_data(
            num_samples=num_samples,
            num_features=num_features,
            random_seed=seed
        )

        true_att = compute_true_att(data)
        est_att = ipw_estimator(data)

        true_atts.append(true_att)
        estimated_atts.append(est_att)

    return true_atts, estimated_atts


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

    num_trials = 50
    true_atts, estimated_atts = monte_carlo(num_trials=num_trials)

    print(f"Mean True ATT: {np.mean(true_atts):.4f}")
    print(f"Mean Estimated ATT: {np.mean(estimated_atts):.4f}")

    plot_results(true_atts, estimated_atts)


if __name__ == "__main__":
    main()
