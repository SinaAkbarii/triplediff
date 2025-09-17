# -*- coding: utf-8 -*-
"""
    This module contains functions to generate synthetic data for testing and demonstration purposes.
"""
import numpy as np
import pandas as pd
from scipy.special import expit


""" Softmax function for the treatment assignment """


def softmax(x):
    """
    Softmax function to convert logits to probabilities.
    :param x: input array
    :return: softmax probabilities
    """
    # clip the input
    # print(f'unclipped range of values: {np.min(x)} - {np.max(x)}')
    clipped_x =  x #np.clip(x, -10, 10)
    # print(f'clipped range of values: {np.min(clipped_x)} - {np.max(clipped_x)}')
    e_x = np.exp(clipped_x - np.max(clipped_x, axis=1, keepdims=True))  # for numerical stability
    return e_x / e_x.sum(axis=1, keepdims=True)  # normalize along the rows


"""
    Two linear regression functions, one for the outcome model, and the other for the propensity score model.
"""


def outcome_model(X, U, G, D, T, coeff=None, observed_propensities=None, random_seed=42):
    """
    Linear regression function for the outcome model.
    :param X: covariate matrix
    :param U: latent variable
    :param G: treatment assignment
    :param D: domain assignment
    :param T: time assignment
    :param coeff: coefficients of the linear model. The default is [210, 27.4, 13.7, 13.7, 13.7],
    taken from Kang & Schafer (2007).
    :return: outcome
    """
    np.random.seed(random_seed)
    if coeff is None:
        coeff = [210, 27.4] + [13.7] * 3
    if len(coeff) != X.shape[1] + 1:
        raise ValueError("The number of coefficients must be equal to the number of features + 1.")
    # if dgp == 'correct' or dgp == 'ps_misspec':
    # h_X = coeff[0] + np.dot(X, coeff[1:])
    # elif dgp == 'or_misspec' or dgp == 'both_misspec':
    X_interaction = np.column_stack([X, np.clip(X[:, 1]**2, -10,10), np.clip(X[:, 3]**2, -10,10)])  # , X[:,0] * X[:, 3], 2*(X[:, 0] * X[:, 2])])
    coeff_interaction = np.concatenate((coeff, [-13.7, 20]))
    h_X = coeff_interaction[0] + np.dot(X_interaction, coeff_interaction[1:])
    # else:
    #     raise ValueError(f"Data generating process '{dgp}' is not supported.")

    # Define the transformation mapping
    transform_map = {
        0: 1,
        1: 2,
        2: 2,
        3: 4,
        4: 1,
        5: 3,
        6: 2,
        7: 5
    }
    vectorized_transform = np.vectorize(transform_map.get)
    X_weights = 0.1 * vectorized_transform(4 * G + 2 * D + T)
    if observed_propensities is None:
        observed_propensities = estimate_p_GD_given_X_batch(X, default=True)
    # observed_propensities is a matrix of shape (n_samples, 4). Of each row, we keep only the column corresponding to
    GD = 2 * G + D
    # Select the column corresponding to GD
    U_weights = 2 * (-1) ** (G + D) * observed_propensities[np.arange(observed_propensities.shape[0]), GD.astype(int)]
    return X_weights * h_X + U_weights * U + np.random.normal(0, 0.5, size=X.shape[0]), observed_propensities


def lr_propensity(X, U, default=None, coeff=None):
    """
    Linear regression function for the propensity score model. This is going to be used to generate the propensity score
    for the treatment assignment based on an expit function.
    :param X: covariate matrix
    :param U: latent variable
    :param default: one of the four default values for the coefficients (one of them is
    taken from Kang & Schafer (2007).
    :param coeff: the coefficients for the linear regression model. if not specified, the default value
    will be used
    :return: the weighted average of covariates.
    """
    if coeff is None:
        if default is None:
            raise ValueError("Coefficients not specified")
        elif default == '00':  # for p(G=0, D=0 | X)
            coeff = np.zeros(5)
        elif default == '01':  # for p(G=0, D=1 | X)
            coeff = .75 * np.array([1, -.5, 0.5, -0.25, -0.1])
        elif default == '10':  # for p(G=1, D=0 | X)
            coeff = .75 * np.array([-.5, .5, -.5, .5, 0.5])
        elif default == '11':  # for p(G=1, D=1 | X)
            coeff = .75 * np.array([0, -.2, 0.5, -0.2, -0.1])
        else:
            raise ValueError(f"Default value '{default}' for lr_propensity is not supported.")
    if len(coeff) != X.shape[1] + 1:
        raise ValueError("The number of coefficients must be equal to the number of features + 1.")
    # Add interaction terms
    X_interaction = np.column_stack([X, X[:, 0] **2, X[:, 3]**2])  # np.abs(X[:, 0] * X[:, 2]), np.abs(X[:, 1])])
    coeff_interaction = np.concatenate((coeff, [-3, 4*coeff[1]] ))
    return coeff_interaction[0] + np.dot(X_interaction, coeff_interaction[1:]) + .6 * U

def prob_T(X, G, D, random_seed=42):
    """
    Samples the treatment assignment T based on the covariates X, domain D, and treatment G.
    :param X: covariate matrix
    :param G: treatment assignment
    :param D: domain assignment
    :param random_seed: random seed for reproducibility
    :return: sampled treatment assignment T
    """
    log_odds_T = 0.1 * X[:, 3] + 0.15 * (D - np.mean(D)) + 0.1 * (G - np.mean(G))
    t_probs = expit(log_odds_T)
    return t_probs

def estimate_p_GD_given_X_batch(X, default=True, coeffs=None, num_mc_samples=1000, random_seed=42, panel=True):
    """
    Function to generate the observed propensity score.
    :param X: covariate matrix
    :param default: if True, the four default values are used as the coefficients (one of them is
    taken from Kang & Schafer (2007).
    :param coeffs: the coefficients for the linear regression model. if not specified, the default values
    will be used
    :param num_mc_samples: number of Monte Carlo samples to use for estimating the propensity score
    :return: the observed propensity score.
    """
    np.random.seed(random_seed)
    # generate the latent variable

    n_samples, n_features = X.shape

    # Sample U: shape (num_mc_samples,)
    U_samples = np.random.normal(0, 1, size=num_mc_samples)

    # Expand X and U to shape (n_samples, num_mc_samples, n_features) and (n_samples, num_mc_samples)
    X_rep = np.repeat(X[:, None, :], num_mc_samples, axis=1)  # (n_samples, num_mc_samples, n_features)
    U_rep = np.repeat(U_samples[None, :], n_samples, axis=0)  # (n_samples, num_mc_samples)

    # Flatten for batch processing: shape (n_samples * num_mc_samples, n_features)
    X_flat = X_rep.reshape(-1, n_features)
    U_flat = U_rep.flatten()

    # Compute logits for each GD configuration
    fl_00 = np.zeros_like(U_flat)
    if default:
        fl_01 = lr_propensity(X_flat, U_flat, default='01')
        fl_10 = lr_propensity(X_flat, U_flat, default='10')
        fl_11 = lr_propensity(X_flat, U_flat, default='11')
    else:
        fl_01 = lr_propensity(X_flat, U_flat, coeff=coeffs[0])
        fl_10 = lr_propensity(X_flat, U_flat, coeff=coeffs[1])
        fl_11 = lr_propensity(X_flat, U_flat, coeff=coeffs[2])

    logits = np.stack([fl_00, fl_01, fl_10, fl_11], axis=1)  # shape: (n_samples * num_mc_samples, 4)
    probs = softmax(logits)  # shape: (n_samples * num_mc_samples, 4)
    # print(probs)
    # Reshape back and average over U samples
    probs_reshaped = probs.reshape(n_samples, num_mc_samples, 4)  # shape: (n_samples, num_mc_samples, 4)
    estimated_probs = probs_reshaped.mean(axis=1)  # shape: (n_samples,)

    # if panel, return estimated_probs as is
    if panel:
        return estimated_probs
    else:  # if cross-sectional, return the probabilities for G,D,T
        joint_probs = np.zeros((n_samples, 8))
        for g in range(2):
            for d in range(2):
                gd_index = 2 * g + d
                p_GD_X = estimated_probs[:, gd_index]
                p_T1_given_GD_X = prob_T(X, d, g)
                joint_probs[:, gd_index * 2] = p_GD_X * (1 - p_T1_given_GD_X)
                joint_probs[:, gd_index * 2 + 1] = p_GD_X * p_T1_given_GD_X
        return joint_probs




"""
    A function to generate the covariate matrix. For now, it is a matrix of independent standard normal variables.
"""


def generate_covariates(num_samples=1000, num_features=4, random_seed=42):
    """
    Generates a covariate matrix of independent standard normal variables.
    :param num_samples: number of samples
    :param num_features: number of features
    :param random_seed: random seed for reproducibility
    :return: a pandas DataFrame containing the covariate matrix
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)

    # Generate the covariate matrix
    covariates = np.random.normal(0, 1, size=(num_samples, num_features + 1))

    return covariates


""" A function to sample treatment assignment values for given covariate values """


def sample_treatments(X, U, coeffs=None, random_seed=42):
    """
    Samples treatment assignment values for given covariate values.
    :param X: covariate matrix
    :param U: latent variable
    :param coeffs: list of coefficients for the propensity regression model
    :return: an array containing the treatment assignment values
    """
    np.random.seed(random_seed)
    n = X.shape[0]  # number of samples

    # Generate propensity scores
    fl_00 = np.zeros(n)
    if coeffs is None:
        fl_01 = lr_propensity(X, U, default='01')
        fl_10 = lr_propensity(X, U, default='10')
        fl_11 = lr_propensity(X, U, default='11')
    else:
        fl_01 = lr_propensity(X, U, default='01', coeff=coeffs[0])
        fl_10 = lr_propensity(X, U, default='10', coeff=coeffs[1])
        fl_11 = lr_propensity(X, U, default='11', coeff=coeffs[2])

    # Compute treatment assignment probabilities according to a softmax function:
    logits = np.stack([fl_00, fl_01, fl_10, fl_11], axis=1)
    p = softmax(logits)
    # Normalize the probabilities
    p = p / np.sum(p, axis=1, keepdims=True)
    # Sample treatment assignment
    u = np.random.rand(n)
    cum_p = np.cumsum(p, axis=1)
    GD = (u[:, None] < cum_p).argmax(axis=1)

    return GD, p
    

def generate_synthetic_data(setting='panel', num_samples=1000, num_features=4, random_seed=42,
                            coeffs_or=None, coeffs_ps=None) -> (pd.DataFrame, list[str], float):
    """
    Generates synthetic data for testing and demonstration purposes.
    :param setting: type of data to generate ('panel' or 'cross-sectional' or 'cross-sectional_ncc')
    :param num_samples: number of samples
    :param num_features: number of features
    :param random_seed: random seed for reproducibility
    :param coeffs_or: coefficients for the outcome regression model
    :param coeffs_ps: a list of coefficients for the propensity score model
    :return: a pandas DataFrame containing the synthetic data
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)

    # Check if the number of features is valid
    if coeffs_or is not None:
        if len(coeffs_or) != num_features + 1:
            raise ValueError("The number of coefficients must be equal to the number of features  + 1.")
        num_features = len(coeffs_or) - 1
    if coeffs_ps is not None:
        if len(coeffs_ps) != num_features + 1:
            raise ValueError("The number of coefficients must be equal to the number of features  + 1.")
        num_features = len(coeffs_ps) - 1

    # Generate covariates
    XU = generate_covariates(num_samples, num_features, random_seed)
    X = XU[:, :-1]  # observed covariates
    U = XU[:, -1]  # unobserved covariate

    # apply some transforms to X according to Kang and Schafer (2007)
    Z_bar = np.vstack(np.array([np.exp(0.5*X[:, 0]), 10+X[:, 1]/(np.exp(X[:, 0])), (0.6 + X[:, 0] * X[:, 2]/25) ** 3,
                                (20+X[:, 1] + X[:, 3]) ** 2]).T)
    # normalize the columns of Z_bar. Z[:, 0] = (Z_bar[:, 0] - np.mean(Z_bar[:, 0]) / np.std(Z_bar[:, 0])
    Z = (Z_bar - np.mean(Z_bar, axis=0)) / np.std(Z_bar, axis=0)
    X = np.clip(Z, -5, 5)

    # Generate domain and treatment assignment
    GD, _ = sample_treatments(X, U, coeffs_ps, random_seed=random_seed)  # GD is a vector of shape (num_samples,)
    D = GD % 2  # Domain assignment
    G = GD // 2  # Treatment assignment

    # Generate the potential outcomes
    # Y_0(0)
    Y_0_0, obs_prop = outcome_model(X, U, G, D, T=0, coeff=None, random_seed=random_seed)
    # Y_1(0)
    Y_1_0, _ = outcome_model(X, U, G, D, T=1, coeff=None, observed_propensities=obs_prop, random_seed=random_seed)

    # Y_1(1)
    Y_1_1 = np.random.normal(0, 1, size=num_samples) + np.mean(Y_1_0[GD == 3]) + 10

    # Create a pandas DataFrame to store the data
    # each column of X is a feature
    covariate_names = [f'X{i}' for i in range(num_features)]
    data = pd.DataFrame(X[:, :], columns=covariate_names)
    # add the treatment assignment and potential outcomes
    data['D'] = D
    data['G'] = G
    if setting == 'panel':
        # Create the observed outcome variables
        Y_0 = Y_0_0
        Y_1 = np.where(D * G == 0, Y_1_0, Y_1_1)
        # add the observed outcome variables to the DataFrame
        data['Y_0'] = Y_0
        data['Y_1'] = Y_1
    elif setting == 'cross-sectional':
        # T is sampled dependent on the covariates, the treatment assignment, and the domain
        t_probs = prob_T(X, G, D, random_seed=random_seed)
        T = np.random.binomial(1, t_probs)
        data['T'] = T
        # Create a single observed outcome variable
        Y = np.where(T == 0, Y_0_0, np.where(D * G == 0, Y_1_0, Y_1_1))
        # add the observed outcome variable to the DataFrame
        data['Y'] = Y
    elif setting == 'cross-sectional_ncc':
        # T is sampled independent of the covariates
        T = np.random.binomial(1, 0.5, size=num_samples)
        data['T'] = T
        # Create a single observed outcome variable
        Y = np.where(T == 0, Y_0_0, np.where(D * G == 0, Y_1_0, Y_1_1))
        # add the observed outcome variable to the DataFrame
        data['Y'] = Y
    else:
        raise ValueError("Setting must be one of the following: 'panel', 'cross-sectional', or 'cross-sectional_ncc'.")

    treatment_effect = Y_1_1 - Y_1_0
    att = np.mean(treatment_effect[GD == 3])
    # counterfactual_outcome, _ = outcome_model(X, U, np.ones(X.shape[0]), np.ones(X.shape[0]), T=1, coeff=None, dgp=dgp,
    #                                        observed_propensities=obs_prop)
    # true_att = np.mean(Y_1_1 - counterfactual_outcome)
    # return the DataFrame
    return data, covariate_names, att, random_seed


