from estimators.dr_estimators import dr_estimator_panel, dr_estimator_rc
from joblib import Parallel, delayed
import os
import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression

class LogisticRegressionWrapper:
    def __init__(self, seed):
        self.model = LogisticRegression(
        solver='lbfgs',              # 'lbfgs' supports multinomial loss
        random_state=seed)

    def fit(self, X, y, id=None):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict_proba(X)  # Return probabilities

def run_trial(data, cnames, models, seed, setting, num_splits, dr_seed, lr, do, hl, bs):
    # print(f'true att: {true_att}')
    if setting == 'cross-sectional' or setting == 'cross-sectional_ncc':
        if models == 'correct':
            est_att, variance, ci_length = dr_estimator_rc(data, models, covariate_names=cnames,
                                                              n_splits=num_splits, seed=dr_seed, id=seed)
        elif models == 'ps_misspec':
            est_att, variance, ci_length = dr_estimator_rc(data, models, covariate_names=cnames,
                                                              ps_model=LogisticRegressionWrapper(dr_seed),
                                                              n_splits=num_splits, seed=dr_seed, id=seed)
        elif models == 'or_misspec':
            est_att, variance, ci_length = dr_estimator_rc(data, models, covariate_names=cnames,
                                                              or_model=Ridge(alpha=1.0),
                                                              n_splits=num_splits, seed=dr_seed, id=seed)
        elif models == 'both_misspec':
            est_att, variance, ci_length = dr_estimator_rc(data, models, covariate_names=cnames,
                                                              ps_model=LogisticRegressionWrapper(dr_seed),
                                                              or_model=Ridge(alpha=1.0), n_splits=num_splits,
                                                              seed=dr_seed, id=seed)
    else:
        if models == 'correct':
            est_att, variance, ci_length = dr_estimator_panel(data, models, covariate_names=cnames,
                                                              n_splits=num_splits, seed=dr_seed, id=seed, lr=lr, do=do,
                                                              hl=hl, bs=bs)
        elif models == 'ps_misspec':
            est_att, variance, ci_length = dr_estimator_panel(data, models, covariate_names=cnames,
                                                              ps_model=LogisticRegressionWrapper(dr_seed),
                                                              n_splits=num_splits, seed=dr_seed, id=seed, lr=lr, do=do,
                                                              hl=hl, bs=bs)
        elif models == 'or_misspec':
            est_att, variance, ci_length = dr_estimator_panel(data, models, covariate_names=cnames,
                                                              or_model=Ridge(alpha=1.0),
                                                              n_splits=num_splits, seed=dr_seed, id=seed, lr=lr, do=do,
                                                              hl=hl, bs=bs)
        elif models == 'both_misspec':
            est_att, variance, ci_length = dr_estimator_panel(data, models, covariate_names=cnames,
                                                              ps_model=LogisticRegressionWrapper(dr_seed),
                                                              or_model=Ridge(alpha=1.0), n_splits=num_splits,
                                                              seed=dr_seed, id=seed, lr=lr, do=do, hl=hl, bs=bs)
    # print(f"time taken for trial {seed + 1}/{num_trials}: {time.time() - start_time:.2f} seconds")
    return est_att, variance, ci_length, seed


def main_sim(dataset, cnames, true_atts, num_trials=1000, num_samples=2000, models='correct', setting='panel', lr=1e-3, do=0.02, hl=64, bs=64):
    # Set parameters
    dr_seed = 42
    num_splits = 3

    est_atts = []
    c_lengths = []
    vars0 = []
    # ids = []

    # Generate synthetic data and run trials in parallel
    max_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))
    # Run all trials in parallel
    futures = Parallel(n_jobs=max_workers, )(
        delayed(run_trial)(dataset[seed].head(num_samples), cnames, models, seed, setting, num_splits, dr_seed, lr, do,
                           hl, bs)
        for seed in range(num_trials)
    )
    for future in futures:
        est_att, variance, ci_length, _ = future  # future.result()
        # true_atts.append(true_att)
        est_atts.append(est_att)
        vars0.append(variance)
        c_lengths.append(ci_length)
        # ids.append(id)
    est_atts = np.array(est_atts)
    vars0 = np.array(vars0)
    c_lengths = np.array(c_lengths)
    est_bias = est_atts - true_atts[:num_trials]
    relat_bias = (est_atts / true_atts) - 1
    is_covered = np.abs(est_bias) < (c_lengths / 2)
    # print(f"avg. ATT:{np.mean(true_atts)}")
    return true_atts, est_atts, est_bias, relat_bias, vars0, c_lengths, is_covered

