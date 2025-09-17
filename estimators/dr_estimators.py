import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import copy
from scipy.stats import norm
import os
import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.dgp import estimate_p_GD_given_X_batch
from estimators.nuisance_estimators import PropensityEstimator, OutcomeEstimator


# Estimator for the panel data setting
def dr_estimator_panel(data: pd.DataFrame, models='correct', covariate_names=None,
                       outcome_names=None, or_model=None, ps_model=None, n_splits=3, seed=42, id=None, lr=1e-3, do=.02, hl=64, bs=64):
    """
    Doubly Robust Estimator for Average Treatment Effect on the Treated (ATT) using K-Fold Cross-fitting.
    :param models: whether the outcome regression and propensity score models are correct or misspecified.
    :param outcome_names: names describing the outcome variable columns
    :param covariate_names: names describing the covariate variable columns
    :param data: pandas DataFrame containing the data
    :param or_model: model for estimating the outcome regression nuisance functions
    :param ps_model: model for estimating the propensity score nuisance functions
    :param n_splits: number of folds for K-Fold Cross-fitting
    :param seed: random seed for reproducibility
    :param id: identifier debugging purposes, can be None
    :return: the ATT estimate
    """

    if outcome_names is None:
        outcome_names = ['Y_0', 'Y_1']

    if covariate_names is None:
        covariate_names = ['X0', 'X1', 'X2', 'X3']
    if models in ['or_misspec', 'both_misspec']:
        covariate_names_or = covariate_names[:1]
    else:
        covariate_names_or = covariate_names
    if models in ['ps_misspec', 'both_misspec']:
        covariate_names_ps = covariate_names[:1]
    else:
        covariate_names_ps = covariate_names

    if len(outcome_names) != 2:
        raise ValueError("Panel data setting requires two outcome measurements per unit: Y_0 and Y_1.")

    # Set random seed for reproducibility
    np.random.seed(seed)

    # Set the default models if not provided.
    if or_model is None:
        or_model = OutcomeEstimator(input_size=2+2*len(covariate_names_or), setting='panel')#, lr=lr, do_rate=do, hidden_layer=hl, batch_size=bs)
    if ps_model is None:
        # ps_model = LogisticRegression(random_state=seed)
        ps_model = PropensityEstimator(input_dim=2*len(covariate_names_ps))#, do_rate=do, hidden_dim=hl, lr=lr, batch_size=bs)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # Initialize the estimates list
    eta_p_estimates = []
    psi_estimates = []
    vars = []

    folds = kf.split(data)

    def est_psi_hat_ell(current_fold):#, next_fold):
        # Get the training data for nuisance functions
        # mask = ~np.isin(current_fold[0], next_fold[1])
        train_idx = current_fold[0]#[mask]
        train_data_mupi = data.iloc[train_idx].copy()
        # Fit the models on the training data. we need to train \pi_{g,d} and \mu_{g,d,\Delta} for
        # every g,d (except g=1,d=1)
        mu_hat = {}
        or_clone = None
        if not isinstance(or_model, OutcomeEstimator):
            for g in [0, 1]:
                for d in [0, 1]:
                    # Fit the outcome regression model
                    if g * d == 1:
                        continue
                    filt = (train_data_mupi['G'] == g) & (train_data_mupi['D'] == d)
                    # if models in ['or_misspec', 'both_misspec']:
                    #     X_train = train_data_mupi.loc[filt, covariate_names[:-1]].values
                    # else:
                    X_train = train_data_mupi.loc[filt, covariate_names_or].values
                    y_train = train_data_mupi.loc[filt, outcome_names[1]] - train_data_mupi.loc[filt, outcome_names[0]]
                    model_clone = copy.deepcopy(or_model)
                    model_clone.fit(X_train, y_train.values)
                    mu_hat[(g, d)] = model_clone
        else:
            train_data = train_data_mupi[(train_data_mupi['G'] == 0) | (train_data_mupi['D'] == 0)]
            X_train = train_data[['G', 'D'] + covariate_names_or].values
            y_train = train_data[outcome_names[1]] - train_data[outcome_names[0]]
            or_clone = copy.deepcopy(or_model)
            or_clone.fit(X_train, y_train.values)

        # Fit the propensity score model
        model_clone = copy.deepcopy(ps_model)
        # if models in ['ps_misspec', 'both_misspec']:
        #     X_train_pi_hat = train_data_mupi[covariate_names[3:]]
        # else:
        X_train_pi_hat = train_data_mupi[covariate_names_ps]
        y_train_pi_hat = 2 * train_data_mupi['G'] + train_data_mupi['D']
        model_clone.fit(X_train_pi_hat.values, y_train_pi_hat.to_numpy(), id=id)

        # estimate 1 / E[G * D]
        train_data_e = data.iloc[current_fold[1]].copy()
        hat_e = len(train_data_e) / np.sum((train_data_e['G'] == 1) & (train_data_e['D'] == 1))

        # Get the test data
        test_data = data.iloc[current_fold[1]].copy()
        X_test_or = test_data[covariate_names_or].values
        X_test_ps = test_data[covariate_names_ps].values
        Y_test = test_data[outcome_names[1]].values - test_data[outcome_names[0]].values
        G_test = test_data['G'].values
        D_test = test_data['D'].values

        # estimate the propensity scores
        predicted_pi_hats = model_clone.predict(X_test_ps)

        def pi_gdr(g, d):  # return the ratio pi_{1,1}(x)/pi_{g,d}(x)
            if g * d == 1:
                return np.ones(len(X_test_ps))
            else:
                n = len(X_test_ps)
                eps = 1 / np.sqrt(n)  # TODO: this should be 1/sqrt(n)
                num = predicted_pi_hats[:, 3]
                denom = predicted_pi_hats[:, 2*g+d]
                return num / np.clip(denom, eps, 1)

        # Compute the estimate for each unit in the test set:
        def estimate_summand(mu_hats, pi_hats, e_hat, Y, G, D, g, d):
            coeff = (-1) ** (g + d + 1)
            # normalized_pi_hats = pi_hats * (G == g) * (D == d) / np.mean(pi_hats * (G == g) * (D == d))
            # return coeff * (e_hat * G * D - normalized_pi_hats) * (Y - mu_hats)  # Hajek
            return coeff * e_hat * (G * D - pi_hats * (G == g) * (D == d)) * (Y - mu_hats)  # Horvitz-Thompson


        eta_p_summands = np.zeros(len(X_test_or))
        for g in [0, 1]:
            for d in [0, 1]:
                if g * d == 1:
                    continue
                if not isinstance(or_model, OutcomeEstimator):
                    # if models in ['or_misspec', 'both_misspec']:
                    #     mu_hats = np.array(mu_hat[(g, d)].predict(X_test[covariate_names].values))
                    # else:
                    mu_hats = np.array(mu_hat[(g, d)].predict(X_test_or))
                else:
                    # mu_hats = np.array(or_clone.predict(np.column_stack([G_test, D_test, X_test])))
                    mu_hats = np.array(or_clone.predict(np.column_stack([g*np.ones(len(G_test)), d*np.ones(len(D_test)),
                                                                         X_test_or])))
                pi_hats = pi_gdr(g, d)
                # Calculate the summands for the current g, d
                eta_p_summands += estimate_summand(mu_hats, pi_hats, 1, Y_test, G_test, D_test, g, d)
        eta_p_estimate = np.mean(eta_p_summands)
        eta_p_estimates.append(eta_p_summands)
        psi_estimates.append(hat_e * eta_p_estimate)
    # Iterate through the K-Folds. First estimate psi_hat
    folds = list(folds)
    folds_copy = copy.copy(folds)
    for fold in folds:
        # while True:
        est_psi_hat_ell(fold)  # Process the current fold

    psi_hat = np.mean(psi_estimates)

    def est_sigma_hat_ell(fold, fold_counter):  # Estimate the variance of psi_hat in the current fold
        train_data = data.iloc[fold[0]].copy()
        test_data = data.iloc[fold[1]].copy()
        G_test = test_data['G'].values
        D_test = test_data['D'].values

        # estimate 1 / E[G * D]
        train_data_e = data.iloc[fold[0]].copy() # different from the estimator version in the sense that here this is a nuisance parameter estimated from the # training data
        hat_e = len(train_data_e) / np.sum((train_data_e['G'] == 1) & (train_data_e['D'] == 1))
        sigma_ell = (hat_e * (eta_p_estimates[fold_counter] - G_test * D_test * psi_hat)) ** 2
        vars.append(np.mean(sigma_ell))


    # now iterate again to estiamte the variance
    # try:
    fold_counter = 0
    for fold in folds_copy:
        # fold = next(folds_copy)
        # while True:
        est_sigma_hat_ell(fold, fold_counter)  # Process the current fold
        fold_counter += 1
    # return the average of the estimates and the length of the confidence intervals:
    variance = np.mean(vars)
    z = norm.ppf(0.975)
    interval_length = 2 * z * np.sqrt(variance) / np.sqrt(len(data))
    return psi_hat, variance, interval_length


# Estimator for the repeated cross-sections setting
def dr_estimator_rc(data: pd.DataFrame, models='correct', covariate_names=None,
                       outcome_names=None, time_name='T', or_model=None, ps_model=None, n_splits=3, seed=42, id=None, non_dichtomous=None):
    """
    Doubly Robust Estimator for Average Treatment Effect on the Treated (ATT) using K-Fold Cross-fitting.
    :param outcome_names: names describing the outcome variable columns
    :param covariate_names: names describing the covariate variable columns
    :param time_name: name of the time variable column
    :param data: pandas DataFrame containing the data
    :param or_model: model for estimating the outcome regression nuisance functions
    :param ps_model: model for estimating the propensity score nuisance functions
    :param n_splits: number of folds for K-Fold Cross-fitting
    :param seed: random seed for reproducibility
    :param id: identifier debugging purposes, can be None
    :param non_dichtomous: a list of indices of non-dichotomous covariates, if any.
    :return: the ATT estimate
    """

    if outcome_names is None:
        outcome_names = 'Y'

    if covariate_names is None:
        covariate_names = ['X0', 'X1', 'X2', 'X3']

    if models in ['or_misspec', 'both_misspec']:
        covariate_names_or = covariate_names[:1]
    else:
        covariate_names_or = covariate_names
    if models in ['ps_misspec', 'both_misspec']:
        covariate_names_ps = covariate_names[:1]
    else:
        covariate_names_ps = covariate_names

    if len(outcome_names) != 1:
        raise ValueError("Repeated cross-sections setting requires one outcome measurement per unit: Y.")

    # Set random seed for reproducibility
    np.random.seed(seed)

    # Set the default models if not provided.
    if or_model is None:
        if non_dichtomous is None:
            non_dichtomous = list(range(len(covariate_names_or)))
        nd = [3+ i for i in non_dichtomous]
        or_model = OutcomeEstimator(input_size=3+len(covariate_names_or)+len(nd), setting='repeated_cross_section', non_dichotomous=nd)
    if ps_model is None:
        # ps_model = LogisticRegression(random_state=seed)
        if non_dichtomous is None:
            non_dichtomous = list(range(len(covariate_names_ps)))
        ps_model = PropensityEstimator(input_dim=len(covariate_names_ps)+len(non_dichtomous), out_dim=8, non_dichotomous=non_dichtomous)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # Initialize the estimates list
    eta_cs_estimates = []
    psi_estimates = []
    vars = []

    folds = kf.split(data)

    def est_psi_hat_ell(current_fold):  # , next_fold):
        # Get the training data for nuisance functions
        # mask = ~np.isin(current_fold[0], next_fold[1])
        train_idx = current_fold[0]#[mask]
        train_data_mupi = data.iloc[train_idx].copy()
        # Fit the models on the training data. we need to train \pi_{g,d} and \mu_{g,d,\Delta} for
        # every g,d (except g=1,d=1)
        mu_hat = {}
        or_clone = None
        if not isinstance(or_model, OutcomeEstimator):
            for g in [0, 1]:
                for d in [0, 1]:
                    for t in [0, 1]:  # for each time point
                        # Fit the outcome regression model
                        if g * d * t == 1:
                            continue
                        filt = (train_data_mupi['G'] == g) & (train_data_mupi['D'] == d) & (train_data_mupi[time_name] == t)
                        X_train = train_data_mupi.loc[filt, covariate_names_or].values
                        y_train = train_data_mupi.loc[filt, outcome_names]
                        model_clone = copy.deepcopy(or_model)
                        model_clone.fit(X_train, y_train.values)
                        mu_hat[(g, d, t)] = model_clone
        else:
            train_data = train_data_mupi[(train_data_mupi['G'] == 0) | (train_data_mupi['D'] == 0) | (train_data_mupi[time_name] == 0)]
            X_train = train_data[['G', 'D', time_name] + covariate_names_or].values
            y_train = train_data[outcome_names]
            or_clone = copy.deepcopy(or_model)
            or_clone.fit(X_train, y_train.values)

        # Fit the propensity score model
        model_clone = copy.deepcopy(ps_model)
        X_train_pi_hat = train_data_mupi[covariate_names_ps]
        y_train_pi_hat = 4 * train_data_mupi['G'] + 2 * train_data_mupi['D'] + train_data_mupi[time_name]
        model_clone.fit(X_train_pi_hat.values, y_train_pi_hat.to_numpy(), id=id)

        # estimate 1 / E[G * D]
        train_data_e = data.iloc[current_fold[1]].copy()
        hat_e = len(train_data_e) / np.sum((train_data_e['G'] == 1) & (train_data_e['D'] == 1) & (train_data_e[time_name] == 1))

        # Get the test data
        test_data = data.iloc[current_fold[1]].copy()
        X_test_or = test_data[covariate_names_or].values
        X_test_ps = test_data[covariate_names_ps].values
        Y_test = test_data[outcome_names].values
        G_test = test_data['G'].values
        D_test = test_data['D'].values
        T_test = test_data[time_name].values


        # estimate the propensity scores
        predicted_pi_hats =  model_clone.predict(X_test_ps)

        def pi_gdtr(g, d, t):  # return the ratio pi_{1,1}(x)/pi_{g,d}(x)
            if g * d * t == 1:
                return np.ones(len(X_test_ps))
            else:
                n = len(X_test_ps)
                eps = 1 / np.sqrt(n)  # TODO: this should be 1/sqrt(n)
                num = predicted_pi_hats[:, 7]
                denom = predicted_pi_hats[:, 4 * g + 2 * d + t]
                return num / np.clip(denom, eps, 1)

        # Compute the estimate for each unit in the test set:
        def estimate_summand(mu_hats, pi_hats, e_hat, Y, G, D, T, g, d, t):
            coeff = (-1) ** (g + d + t)
            return coeff * e_hat * (G * D * T - pi_hats * (G == g) * (D == d) * (T == t)) * (Y - mu_hats)  # Horvitz-Thompson

        eta_cs_summands = np.zeros(len(X_test_or))
        for g in [0, 1]:
            for d in [0, 1]:
                for t in [0, 1]:
                    if g * d * t == 1:
                        continue
                    if not isinstance(or_model, OutcomeEstimator):
                        mu_hats = np.array(mu_hat[(g, d, t)].predict(X_test_or))
                    else:
                        mu_hats = np.array(
                            or_clone.predict(np.column_stack([g * np.ones(len(G_test)), d * np.ones(len(D_test)),
                                                                t * np.ones(len(T_test)),
                                                              X_test_or])))
                    pi_hats = pi_gdtr(g, d, t)
                    # Calculate the summands for the current g, d
                    eta_cs_summands += estimate_summand(mu_hats, pi_hats, 1, Y_test, G_test, D_test, T_test, g, d, t)
        eta_cs_estimate = np.mean(eta_cs_summands)
        eta_cs_estimates.append(eta_cs_summands)
        psi_estimates.append(hat_e * eta_cs_estimate)

    folds = list(folds)
    folds_copy = copy.copy(folds)
    # try:
    #     fold = next(folds)
    #     while True:
    for fold in folds:
        est_psi_hat_ell(fold)  # Process the current fold

    psi_hat = np.mean(psi_estimates)

    def est_sigma_hat_ell(fold, fold_counter):  # Estimate the variance of psi_hat in the current fold
        # train_data = data.iloc[fold[0]].copy()
        test_data = data.iloc[fold[1]].copy()
        G_test = test_data['G'].values
        D_test = test_data['D'].values
        T_test = test_data[time_name].values

        # estimate 1 / E[G * D]
        train_data_e = data.iloc[fold[
            0]].copy()  # different from the estimator version in the sense that here this is a nuisance parameter estimated from the # training data
        hat_e = len(train_data_e) / np.sum((train_data_e['G'] == 1) & (train_data_e['D'] == 1) & (train_data_e[time_name] == 1))
        sigma_ell = (hat_e * (eta_cs_estimates[fold_counter] - G_test * D_test * T_test * psi_hat)) ** 2
        vars.append(np.mean(sigma_ell))

    # now iterate again to estiamte the variance
    # try:
    fold_counter = 0
    for fold in folds_copy:
        # while True:
        est_sigma_hat_ell(fold, fold_counter)  # Process the current fold
        fold_counter += 1
    variance = np.mean(vars)
    z = norm.ppf(0.975)
    interval_length = 2 * z * np.sqrt(variance) / np.sqrt(len(data))
    return psi_hat, variance, interval_length