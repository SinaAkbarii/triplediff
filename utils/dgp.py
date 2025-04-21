# -*- coding: utf-8 -*-
"""
    This module contains functions to generate synthetic data for testing and demonstration purposes.
"""
import numpy as np
import pandas as pd

"""
    Two linear regression functions, one for the outcome model, and the other for the propensity score model.
"""


def lr_outcome(X, coeff=[210, 27.4]+[13.7]*3):
    """
    Linear regression function for the outcome model.
    :param X: covariate matrix
    :param coeff: coefficients of the linear model. The default is [210, 27.4, 13.7, 13.7, 13.7],
    taken from Kang & Schafer (2007).
    :return: outcome
    """
    if len(coeff) != X.shape[1] + 1:
        raise ValueError("The number of coefficients must be equal to the number of features + 1.")
    return coeff[0] + np.dot(X, coeff[1:])


def lr_propensity(X, default=None, coeff=None):
    """
    Linear regression function for the propensity score model. This is going to be used to generate the propensity score
    for the treatment assignment based on an expit function.
    :param X: covariate matrix
    :param default: one of the four default values for the coefficients
    :param coeff: the coefficiants for
    taken from Kang & Schafer (2007).
    :return: propensity score
    """
    coeff = 0.75 * np.array([0, -1, 0.5, -0.25, -1])
    if len(coeff) != X.shape[1] + 1:
        raise ValueError("The number of coefficients must be equal to the number of features + 1.")
    return coeff[0] + np.dot(X, coeff[1:])


"""
    A function to generate the covariate matrix. For now, it is a matrix of independent standard normal variables.
"""


def generate_covariates(num_samples=1000, num_features=4, random_seed=42) -> pd.DataFrame:
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

    return covariates.T


def generate_synthetic_data(num_samples=1000, num_features=4, random_seed=42, coeffs_or=None, coeffs_ps=None) ->\
        pd.DataFrame:
    """
    Generates synthetic data for testing and demonstration purposes.
    :param num_samples: number of samples
    :param num_features: number of features
    :param random_seed: random seed for reproducibility
    :param coeffs_or: coefficients for the outcome regression model
    :param coeffs_ps: coefficients for the propensity score model
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



    # Create the DataFrame
    data = pd.DataFrame({
        'X1': x1,
        'X2': x2,
        'X3': x3,
        'G': G,
        'D': D,
        'Y_0': Y_0,
        'Y_1': Y_1,
        'Y_1_0': Y_1_0,
        'Y_1_1': Y_1_1
    })

    # return the DataFrame
    return data


