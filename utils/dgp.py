# -*- coding: utf-8 -*-
"""
    This module contains functions to generate synthetic data for testing and demonstration purposes.
"""
import numpy as np
import pandas as pd


""" Softmax function for the treatment assignment """


def softmax(x):
    """
    Softmax function to convert logits to probabilities.
    :param x: input array
    :return: softmax probabilities
    """
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # for numerical stability
    return e_x / e_x.sum(axis=1, keepdims=True)  # normalize along the rows


"""
    Two linear regression functions, one for the outcome model, and the other for the propensity score model.
"""


def lr_outcome(X, coeff=None):
    """
    Linear regression function for the outcome model.
    :param X: covariate matrix
    :param coeff: coefficients of the linear model. The default is [210, 27.4, 13.7, 13.7, 13.7],
    taken from Kang & Schafer (2007).
    :return: outcome
    """
    if coeff is None:
        coeff = [210, 27.4] + [13.7] * 3
    if len(coeff) != X.shape[1] + 1:
        raise ValueError("The number of coefficients must be equal to the number of features + 1.")
    return coeff[0] + np.dot(X, coeff[1:])


def lr_propensity(X, default=None, coeff=None):
    """
    Linear regression function for the propensity score model. This is going to be used to generate the propensity score
    for the treatment assignment based on an expit function.
    :param X: covariate matrix
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
            coeff = 0.75 * np.array([0, -1, 0.5, -0.25, -1])
        elif default == '10':  # for p(G=1, D=0 | X)
            coeff = 0.75 * np.array([0, 1, -0.25, 0.75, -1])
        elif default == '11':  # for p(G=1, D=1 | X)
            coeff = 0.75 * np.array([0, -1, 0.5, -0.5, 1])
        else:
            raise ValueError("Default value for lr_propensity not supported.")
    if len(coeff) != X.shape[1] + 1:
        raise ValueError("The number of coefficients must be equal to the number of features + 1.")
    return coeff[0] + np.dot(X, coeff[1:])


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
    covariates = np.random.normal(0, 1, size=(num_samples, num_features))

    return covariates


""" A function to sample treatment assignment values for given covariate values """


def sample_treatments(X, coeffs=None):
    """
    Samples treatment assignment values for given covariate values.
    :param X: covariate matrix
    :param coeffs: list of coefficients for the propensity regression model
    :return: an array containing the treatment assignment values
    """

    n = X.shape[0]  # number of samples

    # Generate propensity scores
    fl_00 = np.zeros(n)
    if coeffs is None:
        fl_01 = lr_propensity(X, default='01')
        fl_10 = lr_propensity(X, default='10')
        fl_11 = lr_propensity(X, default='11')
    else:
        fl_01 = lr_propensity(X, default='01', coeff=coeffs[0])
        fl_10 = lr_propensity(X, default='10', coeff=coeffs[1])
        fl_11 = lr_propensity(X, default='11', coeff=coeffs[2])

    # Compute treatment assignment probabilities according to a softmax function:
    p = softmax(np.array([fl_00, fl_01, fl_10, fl_11]).T)
    # Sample treatment assignment
    GD = np.array([np.random.choice(np.arange(0, 4), p=p[i]) for i in range(n)])

    return GD
    

def generate_synthetic_data(num_samples=1000, num_features=4, random_seed=42, coeffs_or=None, coeffs_ps=None) ->\
        pd.DataFrame:
    """
    Generates synthetic data for testing and demonstration purposes.
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
    X = generate_covariates(num_samples, num_features, random_seed)

    # Generate domain and treatment assignment
    GD = sample_treatments(X, coeffs_ps)
    D = GD % 2  # Domain assignment
    G = GD // 2  # Treatment assignment

    # Generate the potential outcomes
    # Y_0(0)
    Y_0_0 = (1 + G) * lr_outcome(X, coeff=coeffs_or) + np.random.normal(0, 1, size=num_samples) + \
            np.random.normal(0, 1, size=num_samples)
    # Y_1(0)
    Y_1_0 = (D + 2 * (1 + G)) * lr_outcome(X, coeff=coeffs_or) + np.random.normal(0, 1, size=num_samples) + \
            np.random.normal(0, 1, size=num_samples)
    # Y_1(1)
    Y_1_1 = (D + 2 * (1 + G)) * lr_outcome(X, coeff=coeffs_or) + np.random.normal(0, 1, size=num_samples) + \
            np.random.normal(0, 1, size=num_samples)

    # Create the outcome variables
    Y_0 = Y_0_0
    Y_1 = np.where(D * G == 0, Y_1_0, Y_1_1)
    print(np.sum(D * G == 1))

    # Create a pandas DataFrame to store the data
    # each column of X is a feature
    data = pd.DataFrame(X, columns=[f'X{i}' for i in range(num_features)])
    # add the treatment assignment and potential outcomes
    data['D'] = D
    data['G'] = G
    data['Y_0'] = Y_0
    data['Y_1'] = Y_1

    # return the DataFrame
    return data


