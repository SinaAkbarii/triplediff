import sys
import os
import numpy as np
# make sure relative imports work
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.dgp import generate_synthetic_data
import pickle
from joblib import Parallel, delayed
import torch
import random
from util_simulate import main_sim



# Workaround for the issue with debug: add -Xfrozen_modules=off to interpreter options.


if __name__ == "__main__":
    random_state = 42
    tgen = torch.Generator()
    tgen.manual_seed(random_state)
    torch.manual_seed(random_state)
    random.seed(random_state)
    np.random.seed(random_state)

    setting = 'cross-sectional'
    # setting = 'panel'

    num_features = 4
    sample_sizes = [10000]  # 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000,

    input_dir = '../data/synthetic/'
    dataset_file = os.path.join(input_dir, 'dataset_rc.pkl')
    true_atts_file = os.path.join(input_dir, 'true_atts_rc.pkl')
    # if the dataset file already exists, read it
    if os.path.exists(dataset_file) and os.path.exists(true_atts_file):
        print('Loading existing dataset and true atts from files...')
        with open(dataset_file, 'rb') as fp:
            dataset = pickle.load(fp)
        with open(true_atts_file, 'rb') as fp:
            true_atts = pickle.load(fp)
        num_trials = len(dataset)
        cnames = [col for col in dataset[0].columns if col not in ['Y', 'G', 'D', 'T']]
        print(f"Loaded dataset with {num_trials} trials and {len(cnames)} covariates.")
        print('cnames:', cnames)
    else:
        num_trials = 1000
        dataset = {}
        true_atts = np.zeros(num_trials)

        # Generate synthetic data and run trials in parallel
        max_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))
        # Run all trials in parallel
        futures = Parallel(n_jobs=max_workers, )(
            delayed(generate_synthetic_data)(setting=setting, num_samples=max(sample_sizes), num_features=4, random_seed=seed)
            for seed in range(num_trials)
        )
        for future in futures:
            data, cnames, true_att, r_seed = future  # future.result()
            # true_atts.append(true_att)
            dataset[r_seed] = data
            true_atts[r_seed] = true_att

        # store dataset and true_atts via pickle
        with open(dataset_file, 'wb') as fp:
            pickle.dump(dataset, fp)
        with open(true_atts_file, 'wb') as fp:
            pickle.dump(true_atts, fp)

    results = {}
    for num_samples in sample_sizes:
        print(f"Sample size: {num_samples}\n")
        results[num_samples] = {}



        for models in ['correct', 'or_misspec', 'ps_misspec', 'both_misspec']:
            true_atts, estimated_atts, bias, rel_bias, variances, ci_lengths, covered = \
                main_sim({i: dataset[i] for i in range(num_trials)}, cnames, true_atts[:num_trials], num_trials, num_samples, models, setting)#, lr=1e-3, do=0.02, hl=64, bs=64)
            print(f"Running with model: {models}, sample size: {num_samples}")
            print(f"mean:{np.mean(rel_bias)}")
            print(f"median:{np.median(rel_bias)}")
            print()
            results[num_samples][models] = {
                'true_atts': list(true_atts),
                'estimated_atts': list(estimated_atts),
                'bias': list(bias),
                'rel_bias': list(rel_bias),
                'variances': list(variances),
                'ci_lengths': list(ci_lengths),
                'covered': list(covered.astype(int))
            }
            output_dir = '../results/synthetic/'
            results_file = os.path.join(output_dir, 'rc_' + str(int(sample_sizes[0]/1000)) + 'k.pkl')
            with open(results_file, 'wb') as fp:
                pickle.dump(results, fp)



