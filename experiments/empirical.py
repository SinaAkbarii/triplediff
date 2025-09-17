import pandas as pd
import pickle
from sklearn.linear_model import Ridge, LogisticRegression

def stratified_bootstrap(data, group_cols, random_state=None):
    """
    Stratified bootstrap sampling based on specified group columns.
    data: pd.DataFrame Input data to bootstrap.
    group_cols: list Columns to define strata, e.g. ['G', 'D', 'T'].
    random_state : int or None Random seed for reproducibility.
    Returns pd.DataFrame Bootstrapped dataframe with same proportion of strata as original.
    """
    bootstrapped_parts = []
    for _, group_df in data.groupby(group_cols):
        boot_group = group_df.sample(
            n=len(group_df),  # same size as original group
            replace=True,
            random_state=random_state
        )
        bootstrapped_parts.append(boot_group)

    return pd.concat(bootstrapped_parts).reset_index(drop=True)

class LogisticRegressionWrapper:
    def __init__(self, seed):
        self.model = LogisticRegression(
        solver='lbfgs',              # 'lbfgs' supports multinomial loss
        random_state=seed)

    def fit(self, X, y, id=None):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict_proba(X)  # Return probabilities

# with open('../data/maternity application/maternity_mandate.pkl', 'rb') as f:
#     dataset = pickle.load(f)

# calculate the average earnings per hour for experimental states (IL, NJ, NY) and control states (OH, IN, CT, MA, NC) for each T
# report one number for each group: experimental vs control, and T = 0 vs T = 1
def average_earnings_experimental_control(dataset, g):
    # filter for G == 1
    filtered_data = dataset[dataset['G'] == g].copy()
    # Group by experimental/control and T, then calculate the mean of Y
    filtered_data['experimental'] = filtered_data['state_code'].apply(lambda x: 1 if x in [33, 22, 21] else 0)  # IL, NJ, NY
    filtered_data['control'] = filtered_data['state_code'].apply(lambda x: 1 if x in [31, 32, 11, 12, 53] else 0)  # OH, IN, CT, MA, NC
    avg_earnings = filtered_data.groupby(['experimental', 'control', 'T'])['Y'].mean().reset_index()
    avg_earnings['experimental'] = avg_earnings['experimental'].map({1: 'Experimental', 0: 'Control'})
    avg_earnings['control'] = avg_earnings['control'].map({1: 'Control', 0: 'Experimental'})
    avg_earnings = avg_earnings.pivot_table(index='T', columns='experimental', values='Y').reset_index()
    avg_earnings.columns.name = None  # Remove the name of the columns
    avg_earnings.columns = ['T', 'Control', 'Experimental']
    return avg_earnings
# avg_earnings_experimental_control = average_earnings_experimental_control(dataset, 0)



# regression model for triple difference estimation
# import statsmodels.api as sm
# def run_regression(dataset):
#     # Define the independent variables
#     X = dataset[['education', 'age', 'sex', 'marital_status', 'white/non-white', 'union', 'white_collar', 'T', 'D', 'G']].copy()
#     # Add interaction terms
#     X = X.assign(
#         marital_status_sex=X['marital_status'] * X['sex'],
#         G_D=dataset['G'] * dataset['D'],
#         T_D=dataset['T'] * dataset['D'],
#         T_G=dataset['T'] * (dataset['G']),
#         T_G_D= dataset['T'] * dataset['G'] * dataset['D']
#     )
#     # Add a constant term for the intercept
#     X = sm.add_constant(X)
#     # Define the dependent variable
#     y = dataset['Y']
#     # Fit the model
#     model = sm.OLS(y, X).fit()
#     # Print the summary of the model
#     print(model.summary())
# # Run the regression for dataset
# # run_regression(dataset)

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from estimators.dr_estimators import dr_estimator_rc
import numpy as np
import torch
import random
import pickle
from joblib import Parallel, delayed

def run_empirical(data, run_id):
    # bootstrap the data
    # data = data.sample(frac=1, replace=True, random_state=run_id).reset_index(drop=True)
    # data = stratified_bootstrap(data, group_cols=['G', 'D', 'T'], random_state=run_id)
    est_att, variance, ci_length = dr_estimator_rc(data,
                                                   covariate_names=['education', 'age', 'sex', 'marital_status',
                                                                    'white/non-white', 'union', 'white_collar'],
                                                   n_splits=3, seed=42, non_dichtomous=[0, 1, 3])

    return est_att, variance, ci_length, run_id

    # est_att_ps_logistic, variance_ps_logistic, ci_length_ps_logistic = dr_estimator_rc(dataset,
    #                                                                                    covariate_names=['education',
    #                                                                                                     'age', 'sex',
    #                                                                                                     'marital_status',
    #                                                                                                     'white/non-white',
    #                                                                                                     'union',
    #                                                                                                     'white_collar'],
    #                                                                                    n_splits=3, seed=42,
    #                                                                                    non_dichtomous=[0, 1, 3],
    #                                                                                    ps_model=LogisticRegressionWrapper(
    #                                                                                        42))
    #
    # est_att_or_ridge, variance_or_ridge, ci_length_or_ridge = dr_estimator_rc(dataset,
    #                                                                           covariate_names=['education', 'age',
    #                                                                                            'sex', 'marital_status',
    #                                                                                            'white/non-white',
    #                                                                                            'union', 'white_collar'],
    #                                                                           n_splits=3, seed=42,
    #                                                                           non_dichtomous=[0, 1, 3],
    #                                                                           or_model=Ridge(alpha=1.0))


if __name__ == "__main__":
    random_state = 42
    tgen = torch.Generator()
    tgen.manual_seed(random_state)
    torch.manual_seed(random_state)
    random.seed(random_state)
    np.random.seed(random_state)

    # '/scratch/akbari'  #
    input_dir =  '../data/maternity application/'
    dataset_file = os.path.join(input_dir, 'maternity_mandate.pkl')
    with open(dataset_file, 'rb') as f:
        dataset = pickle.load(f)

    max_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))
    num_trials = 1
    # Run all trials in parallel
    futures = Parallel(n_jobs=max_workers, )(
        delayed(run_empirical)(dataset.copy(), run_id) for run_id in range(num_trials)
    )

    est_atts = []
    vars0 = []
    c_lengths = []
    for future in futures:
        est_att, variance, ci_length, _ = future  # future.result()
        est_atts.append(est_att)
        vars0.append(variance)
        c_lengths.append(ci_length)
        print(f"Estimated ATT: {est_att}, CI Length: {ci_length}")

    results = {
        'estimated_atts': est_atts,
        'variances': vars0,
        'ci_lengths': c_lengths
    }
    output_dir = '/scratch/akbari'#'../results/synthetic/'
    results_file = os.path.join(output_dir, 'empirical.pkl')
    with open(results_file, 'wb') as fp:
        pickle.dump(results, fp)






# read from file
with open('results/empirical.pkl', 'rb') as f:
    resemp2 = pickle.load(f)
estimated_atts = resemp2['estimated_atts']

# Q-Q plot to check normality of estimated ATTs
import matplotlib.pyplot as plt
import scipy.stats as stats
def qq_plot(data):
    """
    Generate a Q-Q plot to check the normality of the data.
    data: list or numpy array of data points
    """
    stats.probplot(data, dist="norm", plot=plt)
    plt.title('Q-Q Plot')
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Sample Quantiles')
    plt.grid()
    plt.show()
qq_plot(estimated_atts)

# percentile of the estimated ATTs
np.percentile(estimated_atts, 90)

from scipy.stats import norm
z = norm.ppf(0.89)
np.mean(estimated_atts) + z * np.std(estimated_atts, ddof=1)

# compute the p-value for the estimated ATT list
def compute_p_value(estimated_atts, true_att=0):
    """
    Compute the p-value for the estimated ATT list.
    estimated_atts: list of estimated ATTs
    true_att: true ATT value to test against (default is 0)
    Returns float p-value
    """
    # Convert to numpy array for easier calculations
    estimated_atts = np.array(estimated_atts)
    # Calculate the number of estimates greater than or equal to the true ATT
    count_greater_equal = np.sum(estimated_atts >= 0)
    # Calculate the total number of estimates
    total_estimates = len(estimated_atts)
    # Calculate the p-value
    p_value = count_greater_equal / total_estimates
    return p_value

# compute which quantile of the standard normal distribution corresponds to a given value
def compute_quantile(value, mean=0, std=1):
    """
    Compute the quantile of the standard normal distribution for a given value.
    value: float value to compute quantile for
    mean: float mean of the distribution (default is 0)
    std: float standard deviation of the distribution (default is 1)
    Returns float quantile
    """
    from scipy.stats import norm
    2*(1-norm.cdf(1.2458037224546223, loc=0, scale=1))