import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import KFold
from sklearn.base import clone
from copy import copy


def dr_estimator(data: pd.DataFrame, covariate_names=['X0', 'X1', 'X2', 'X3'],
                 outcome_names=['Y_0', 'Y_1'], or_model=None, ps_model=None, n_splits=4, seed=42):
    """
    Doubly Robust Estimator for Average Treatment Effect on the Treated (ATT) using K-Fold Cross-fitting.
    :param outcome_names: names describing the outcome variable columns
    :param covariate_names: names describing the covariate variable columns
    :param data: pandas DataFrame containing the data
    :param or_model: model for estimating the outcome regression nuisance functions
    :param ps_model: model for estimating the propensity score nuisance functions
    :param n_splits: number of folds for K-Fold Cross-fitting
    :param seed: random seed for reproducibility
    :return: the ATT estimate
    """

    # TODO: Generalize to the repeated cross-sections setting later.
    if len(outcome_names) != 2:
        raise ValueError("Only panel data case implemented so far.")

    # Set random seed for reproducibility
    np.random.seed(seed)

    # Set the default models if not provided.
    if or_model is None:
        or_model = Ridge(alpha=1.0, random_state=seed)
    if ps_model is None:
        ps_model = LogisticRegression(random_state=seed)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # Initialize the estimates list
    estimates = []

    folds = kf.split(data)

    def iter_fold(current_fold, next_fold):
        # Get the training data for nuisance functions
        train_idx = np.array(list(set(current_fold[0]).difference(next_fold[1])))
        train_data_mupi = data.iloc[train_idx].copy()
        # Fit the models on the training data. we need to train \pi_{g,d} and \mu_{g,d,\Delta} for
        # every g,d (except g=1,d=1)
        mu_hat = {}
        pi_hat = {}
        for g in [0, 1]:
            for d in [0, 1]:
                filt = (train_data_mupi['G'] == g) & (train_data_mupi['D'] == d)
                # Fit the propensity score model
                model_clone = clone(ps_model)
                model_clone.fit(train_data_mupi[covariate_names], filt.astype(int))
                # print the coefficients
                # print("Propensity score model coefficients for G={}, D={}: {}".format(g, d, model_clone.coef_))
                pi_hat[(g, d)] = model_clone
                # Fit the outcome regression model
                if g == 1 and d == 1:
                    continue
                X_train = train_data_mupi.loc[filt, covariate_names]
                y_train = train_data_mupi.loc[filt, outcome_names[1]] - train_data_mupi.loc[filt, outcome_names[0]]
                model_clone = clone(or_model)
                model_clone.fit(X_train, y_train)
                # print the coefficients
                # print("Outcome regression model coefficients for G={}, D={}: {}".format(g, d, model_clone.coef_))
                mu_hat[(g, d)] = model_clone

        # Calculate the estimates for the test data
        def mu_delta(x, g, d):
            x_df = pd.DataFrame([x], columns=covariate_names)
            return mu_hat[(g, d)].predict(x_df)

        def pi_gdr(x, g, d):  # return the ratio pi_{1,1}(x)/pi_{g,d}(x)
            x_df = pd.DataFrame([x], columns=covariate_names)
            if g == 1 and d == 1:
                return 1
            else:
                return pi_hat[(1, 1)].predict_proba(x_df)[:, 1][0] / pi_hat[(g, d)].predict_proba(x_df)[:, 1][0]

        # Compute the estimate for each unit in the test set:
        def estimate(x, y, G, D):
            total = 0.0
            for g in [0, 1]:
                for d in [0, 1]:
                    if g == 1 and d == 1:
                        continue
                    coeff = (-1) ** (g + d + 1)
                    total += coeff * (G * D - pi_gdr(x, g, d) * (G == g) * (D == d)) * (y - mu_delta(x, g, d))
            return total

        # Estimate \hat{e} from a separate fold of the data:
        train_data_e = data.iloc[next_fold[1]].copy()
        hat_e = np.sum((train_data_e['G'] == 1) & (train_data_e['D'] == 1)) / len(train_data_e)
        print("hat_e: {}".format(hat_e))
        # Estimate the ATT:
        test_data = data.iloc[current_fold[1]].copy()
        X_test = test_data[covariate_names]
        Y_test = test_data[outcome_names[1]] - test_data[outcome_names[0]]
        G_test = test_data['G']
        D_test = test_data['D']
        att_estimate = \
            np.mean(
                hat_e * np.array([estimate(x, y, G, D) for x, y, G, D in zip(X_test.values, Y_test.values, G_test.values,
                                                                    D_test.values)]))
        estimates.append(att_estimate)

    # Iterate through the K-Folds
    try:
        fold_1 = next(folds)
        current_fold = copy(fold_1)
        try:
            next_fold = next(folds)
            num_stops = 0
        except StopIteration:
            raise ValueError("Only one fold available.")
        while True:
            iter_fold(current_fold, next_fold)  # Train on the current fold, and estimate using the test fold
            # Move to the next fold
            try:
                current_fold = next_fold
                next_fold = next(folds)
            except StopIteration:  # the first time this happens, we use the first fold as next fold. the second time,
                # we are done.
                if num_stops == 0:
                    num_stops += 1
                    next_fold = copy(fold_1)
                else:
                    break
        # return the average of the estimates:
        return np.mean(estimates)

    except StopIteration:
        raise ValueError("No folds available. Check your data and n_splits.")
