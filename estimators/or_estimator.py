import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.base import clone


def or_estimator(data: pd.DataFrame, model=None, n_splits=4, seed=42):
    # Set random seed for reproducibility
    np.random.seed(seed)

    if model is None:
        model = Ridge(alpha=1.0, random_state=seed)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    estimates = []

    for train_index, test_index in kf.split(data):
        train_data = data.iloc[train_index].copy()
        test_data = data.iloc[test_index].copy()

        # Fit the models on the training data. we need to train \mu_{g,d,\Delta} for every g,d
        mu_hat = {}

        for g in [0, 1]:
            for d in [0, 1]:
                if g == 1 and d == 1:
                    continue
                filt = (train_data['G'] == g) & (train_data['D'] == d)
                X_train = train_data.loc[filt, ['X1', 'X2', 'X3']]
                y_train = train_data.loc[filt, 'Y_1'] - train_data.loc[filt, 'Y_0']
                model_clone = clone(model)
                model_clone.fit(X_train, y_train)
                mu_hat[(g, d)] = model_clone

        # Calculate the estimates for the test data
        def mu_delta(x, g, d):
            feature_names = ['X1', 'X2', 'X3']
            x_df = pd.DataFrame([x], columns=feature_names)
            return mu_hat[(g, d)].predict(x_df)

        # Estimate the ATT:
        held_out = test_data.loc[(test_data['G'] == 1) & (test_data['D'] == 1)].copy()
        X_held = held_out[['X1', 'X2', 'X3']].values
        Y_held = held_out['Y_1'].values - held_out['Y_0'].values

        mu_01 = np.array([mu_delta(x, 0, 1) for x in X_held])
        mu_10 = np.array([mu_delta(x, 1, 0) for x in X_held])
        mu_00 = np.array([mu_delta(x, 0, 0) for x in X_held])

        E2 = np.mean(mu_01 + mu_10 - mu_00)
        E1 = np.mean(Y_held)

        # Calculate the ATT estimate
        att_estimate = E1 - E2
        estimates.append(att_estimate)
    # Return the average of the estimates
    return np.mean(estimates)
