import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import KFold
from sklearn.base import clone


def ipw_estimator(data: pd.DataFrame, model=None, n_splits=4, seed=42):
    # Set random seed for reproducibility
    np.random.seed(seed)

    if model is None:
        model = LogisticRegression(solver="lbfgs", max_iter=1000)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    estimates = []

    for train_index, test_index in kf.split(data):
        train_data = data.iloc[train_index].copy()
        test_data = data.iloc[test_index].copy()

        # Fit the models on the training data. we need to train \mu_{g,d,\Delta} for every g,d
        pi_hat = {}

        for g in [0, 1]:
            for d in [0, 1]:
                label = (train_data['G'] == g) & (train_data['D'] == d)
                X_train = train_data[['X1', 'X2', 'X3']]
                model_clone = clone(model)
                model_clone.fit(X_train, label.astype(int))
                pi_hat[(g, d)] = model_clone

        # Calculate the estimates for the test data
        def pi_gd(x, g, d):
            feature_names = ['X1', 'X2', 'X3']
            x_df = pd.DataFrame([x], columns=feature_names)
            return pi_hat[(g, d)].predict_proba(x_df)[:, 1][0]
            # return mu_hat[(g, d)].predict(x_df)

        # Compute \hat{\rho}(X,G,D) for each unit
        def rho(x, G, D):
            total = 0.0
            for g in [0, 1]:
                for d in [0, 1]:
                    coeff = (1 - g - G) * (1 - d - D)
                    denom = pi_gd(x, g, d)
                    total += coeff / denom if denom > 0 else 0.0
            return total

        # Compute \hat{e}_p = 1 / E[G * D]
        e_p = 1.0 / np.mean(data['G'] * data['D'])

        # Estimate the ATT:
        held_out = test_data.copy()
        Gs = held_out['G'].values
        Ds = held_out['D'].values
        Xs = held_out[['X1', 'X2', 'X3']].values
        Ys = held_out['Y_1'].values - held_out['Y_0'].values

        # Estimate pi_{1,1}(X)
        pi_11s = [pi_gd(x, 1, 1) for x in held_out[['X1', 'X2', 'X3']].values]

        # Estimate rhos:
        rhos = [rho(Xs[i], Gs[i], Ds[i]) for i in range(len(held_out))]

        weighted_outcomes = e_p * np.array(pi_11s) * np.array(rhos) * Ys

        # Calculate the ATT estimate
        att_estimate = np.mean(weighted_outcomes)
        estimates.append(att_estimate)
    # Return the average of the estimates
    return np.mean(estimates)