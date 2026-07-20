# Semiparametric Triple Differences Estimators

This repository contains the Python code and replication materials for the paper:

> Sina Akbari, Negar Kiyavash, and AmirEmad Ghassami,  
> **“Semiparametric Triple Differences Estimators.”**

The code implements the proposed doubly robust, influence-function-based estimators for:

- panel data; and
- repeated cross-sectional data.

It also contains the simulation designs, the maternity-benefits application, and scripts used to generate numerical summaries and figures.

## Repository structure

```text
triplediff/
├── data/
│   ├── maternity application/
│   │   ├── raw data/                 # Raw May CPS data and related source files
│   │   └── maternity_mandate.pkl     # Processed application data
│   └── synthetic/
│       ├── true_atts_panel.pkl        # Stored true ATT values for panel simulations
│       └── true_atts_rc.pkl           # Stored true ATT values for repeated-CS simulations
├── estimators/
│   ├── dr_estimators.py               # Cross-fitted doubly robust ATT estimators
│   └── nuisance_estimators.py         # PyTorch outcome and propensity-score models
├── experiments/
│   ├── data_preparation.py            # Prepares the maternity-benefits data
│   ├── dr_panel.py                    # Panel-data simulation driver
│   ├── dr_rc.py                       # Repeated-cross-section simulation driver
│   ├── empirical.py                   # Maternity-benefits application
│   └── util_simulate.py               # Shared simulation and parallelization utilities
├── results/
│   ├── empirical.pkl                  # Stored empirical results
│   └── synthetic/                     # Stored simulation results and generated figures
├── utils/
│   └── dgp.py                         # Synthetic data-generating processes
└── visualization.py                   # Figure and table-generation code
```

## Requirements

Python 3.9 or later is recommended.

The required Python packages are:

```text
numpy
pandas
scipy
scikit-learn
torch
joblib
matplotlib
seaborn
pyreadr
```

`statsmodels` is optional and is needed only for the commented-out benchmark regression in `experiments/empirical.py`.

The plotting script enables LaTeX rendering in Matplotlib. A working LaTeX installation with the `fourier` package is therefore needed for publication-style figures. Alternatively, set `text.usetex` to `False` in `visualization.py`.

## Installation

```bash
git clone https://github.com/SinaAkbarii/triplediff.git
cd triplediff

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

python -m pip install --upgrade pip
pip install numpy pandas scipy scikit-learn torch joblib matplotlib seaborn pyreadr
```

The scripts use the environment variable `SLURM_CPUS_PER_TASK` to determine the number of parallel workers. On a local machine, it can be set manually:

```bash
export SLURM_CPUS_PER_TASK=4
```

## Preparing the maternity-benefits data

The raw R data file is read from:

```text
data/maternity application/raw data/cpsmay74-78.RData
```

From the `experiments/` directory, run:

```bash
python data_preparation.py
```

This creates or overwrites:

```text
data/maternity application/maternity_mandate.pkl
```

## Running the simulations

### Panel-data simulations

From `experiments/`:

```bash
PYTHONPATH=.. python dr_panel.py
```

The script:

1. generates or loads synthetic panel datasets;
2. evaluates the four nuisance-model scenarios;
3. stores the resulting estimates, biases, variances, confidence-interval lengths, and coverage indicators in `results/synthetic/`.

### Repeated-cross-section simulations

From `experiments/`:

```bash
PYTHONPATH=.. python dr_rc.py
```

The repeated-cross-section script follows the same workflow.

### Simulation configuration

The main settings are defined near the bottom of `dr_panel.py` and `dr_rc.py`, including:

```python
sample_sizes = [10000]
num_trials = 1000
```

The default neural-network experiments are computationally intensive. For a quick installation check, first use a smaller sample size and a small number of trials. To reproduce results across sample sizes 1,000 through 10,000, use:

```python
sample_sizes = list(range(1000, 11000, 1000))
```

## Running the empirical application


### Point estimate versus bootstrap

With `num_trials = 1` and both resampling lines commented out, the script computes the point estimate once.

To run a bootstrap analysis, the resampling step in `run_empirical` must be enabled and `num_trials` must be changed to the number of bootstrap replications used in the paper. For example, for a stratified bootstrap:

```python
data = stratified_bootstrap(
    data,
    group_cols=["G", "D", "T"],
    random_state=run_id,
)
```

and:

```python
num_trials = 1000
```


## Generating figures and numerical summaries

Run `visualization.py` from the repository root:

```bash
cd ..
python visualization.py
```


## Citation

Please cite the accompanying paper when using this code. 

