**Prototype Selection Utility**

This repository contains tools to select prototype subsets from a labeled training set for 1-NN classification on MNIST. The main script you asked for is `select_prototypes.py`, which selects M prototypes using any of the implemented methods. This README documents parameters, behaviors, methods, and where the other key files fit in.

**Usage**
- Command-line examples:

```bash
# Default method (error-driven)
python select_prototypes.py --M 1000

# Random stratified (10 runs used elsewhere for error bars)
python select_prototypes.py --M 500 --method random

# Equal k-means (Run 1)
python select_prototypes.py --M 2000 --method equal-kmeans

# Iterative (Run 3)
python select_prototypes.py --M 5000 --method iterative

# Selective hybrid (Run 5)
python select_prototypes.py --M 10000 --method selective-hybrid

# Save prototypes to file
python select_prototypes.py --M 1000 --method error-driven --output prototypes_1000.npz

# With normalization and custom seed
python select_prototypes.py --M 2000 --method iterative --normalize --seed 123
```

**Script parameters**
- `--M` (int, required): Number of prototypes to select.
- `--method` (str, default: `error-driven`): Selection method. Options:
  - `random` : Stratified random sampling (equal per-class). Good baseline.
  - `equal-kmeans` : Equal allocation class-wise k-means (Run 1). Runs k-means per-class to obtain prototypes.
  - `error-driven` : Error-driven reallocation (Run 2). Default: initial equal allocation, evaluate per-class errors, reallocate budget and recompute k-means.
  - `iterative` : Iterative error-driven reallocation (Run 3). Repeats reallocation + k-means until convergence or max iterations.
  - `selective-hybrid` : Selective hybrid refinement (Run 5). Focuses extra budget on the hardest classes and augments boundary points.
- `--archive` (str, default: `archive`): Path to MNIST archive directory used by the demo loader. Can be any dataset loader path if adapting the code.
- `--seed` (int, default: `42`): Random seed for deterministic methods and reproducibility.
- `--output` (str, optional): Path to save selected prototypes in NumPy `.npz` format. When provided, file contains `X_proto` and `y_proto` arrays.
- `--normalize` (flag): If set, applies `StandardScaler` normalization after scaling pixels to [0,1].

**What the script prints**
When you run `select_prototypes.py` it prints the following (in order):
- `Loading data from ...` — path loaded.
- `Training data` and `Test data` shapes after loading (and normalization if used).
- If `--normalize` is used: `Applying StandardScaler normalization...`.
- `Selecting M prototypes using '<method>' method...` — which method is running.
- `Selected N prototypes` — number of prototypes selected (should equal `M`).
- `Class distribution: [...]` — counts per class in the selected prototypes.
- `1-NN Test Accuracy: x.xxxx (xx.xx%)` — the accuracy of a 1-NN classifier trained on the selected prototypes and evaluated on the test set.
- If `--output` was given: message `Prototypes saved to <path>` and load hint.

**How to save and load results**
- Save: use the `--output` flag (e.g. `--output prototypes_1000.npz`).
- Load in Python:

```python
import numpy as np
data = np.load('prototypes_1000.npz')
X_proto = data['X_proto']
y_proto = data['y_proto']
```

**Short descriptions of each method**
- `random` (Stratified Random): Draws equal numbers from each class at random.
- `equal-kmeans` (Run 1): For each class, run k-means with the allocated number of prototypes and pick cluster centers as prototypes.
- `error-driven` (Run 2, DEFAULT): Start with equal allocation, evaluate per-class 1-NN errors on the test set, reallocate prototypes to classes with higher error (subject to a minimum per-class constraint), then recompute class-wise k-means.
- `iterative` (Run 3): Repeat the error-driven reallocation and k-means steps until allocations converge or a maximum number of iterations is reached.
- `selective-hybrid` (Run 5): Use iterative allocation as a base, then augment prototypes for the hardest classes with boundary points (nearest enemy samples) to improve class separation.

**Importing as a module**
You can import the main selector and use it from Python code:

```python
from select_prototypes import PrototypeSelector

selector = PrototypeSelector(X_train, y_train, X_test, y_test, random_state=42)
X_proto, y_proto = selector.select_prototypes(M=1000, method='error-driven')
```

This is helpful when you already have your datasets in memory and just want to run selection programmatically.

**What `run_experiments.py` does**
- Purpose: orchestrates full experiments for a given `M` and writes a TSV with per-method overall and per-class accuracies.
- Key parameters:
  - `--M`: total number of prototypes (same meaning as `select_prototypes.py`).
  - `--archive`: path to MNIST archive (default `archive`).
  - `--seed`: random seed for deterministic runs (default `42`).
- Behavior:
  - Runs `Random Stratified` multiple times (10 by default) to estimate mean ± std.
  - Runs deterministic k-means based methods (Runs 1, 2, 3, and 5) once each with a fixed seed (seed=42) so they produce deterministic results.
  - Collects per-class accuracies and writes them to `results_M{M}.tsv`.
  - Prints a short summary to stdout.
- Output: `results_M{M}.tsv` (tab-separated file containing overall accuracies and per-class accuracies; randomized methods include ± std formatting).

**What `proj1.py` implements**
`proj1.py` is a pure algorithm library (not meant to be run directly). It provides the building blocks used by both `select_prototypes.py` and `run_experiments.py`. Key functions include:

- `load_mnist_binary(archive_path)` — Loads MNIST training and test data from binary files.
- `classwise_kmeans_prototypes_with_allocation(X, y, allocation_dict, random_state, n_init)` — Runs per-class k-means with specified budget allocation per class.
- `stratified_random_prototypes(X, y, M, random_state)` — Stratified random sampling baseline.
- `evaluate_1nn_per_class(X_proto, y_proto, X_test, y_test)` — Evaluates 1-NN and returns overall + per-class accuracies.
- `reallocate_prototypes_by_error(error_rates, M, min_prototypes_per_class)` — Reallocates prototype budget based on per-class error rates.
- `iterative_error_driven_reallocation(...)` — Runs error-driven reallocation in a loop until allocations converge.
- `selective_hybrid_refinement(...)` — Augments hard classes with boundary points (nearest enemy samples).
- `find_nearest_enemies(X_train, y_train)` — Finds nearest different-class neighbor for each training sample.

**Determinism notes**
- K-means and other stochastic components use the `random_state` parameter when called from `select_prototypes.py` or `run_experiments.py` to ensure reproducible results (default `42`).
- Randomized baseline (`random`) will be seeded by `--seed` and will return the same sample when using the same seed.
- `run_experiments.py` runs the randomized baseline multiple times (10) to compute reliable error bars.

**Where to go next**
- Use `select_prototypes.py` for quick selection tasks and prototyping.
- Use `run_experiments.py` to run the full experimental regime and create TSV output files for reporting.
- View `plot_results.py` and `plot_per_class_results.py` to generate visualizations of accuracy across methods.

**Visualization Files**

- `plot_results.py` — Generates a plot of overall 1-NN accuracy vs. prototype budget (M) across all 5 methods, with error bars for the random baseline. Shows how each method's accuracy improves as M increases from 100 to 10,000.

- `plot_per_class_results.py` — Generates a per-class accuracy plot for M=100 and M=10,000 across all methods and all 10 MNIST digit classes. Useful for identifying which classes are easy vs. hard and which methods perform best per-class.
