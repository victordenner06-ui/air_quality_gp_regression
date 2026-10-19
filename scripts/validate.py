"""
Local validation of a solution file on a held-out split of the training data.

Usage (from the repository root):
    python scripts/validate.py solution.py
    python scripts/validate.py experiments/solution_local_gp_matern.py --val-frac 0.2 --seed 1

Caveat: the training targets contain measurement noise while the official test
targets do not, so this validation cost is higher than the checker's public
score. Use it to compare models with each other, not to predict the score.
"""
import argparse
import importlib.util
import os
import sys
import time

import numpy as np


def load_solution(path):
    """Import a solution file as a module, whatever its name."""
    path = os.path.abspath(path)
    name = os.path.splitext(os.path.basename(path))[0].replace("+", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description="Validate a PAI Task 1 solution file.")
    parser.add_argument("solution", help="path to a solution file, e.g. solution.py")
    parser.add_argument("--val-frac", type=float, default=0.2, help="fraction of points held out (default 0.2)")
    parser.add_argument("--seed", type=int, default=1, help="seed of the train/validation split (default 1)")
    parser.add_argument("--data-dir", default=".", help="folder containing train_x.csv and train_y.csv")
    args = parser.parse_args()

    sol = load_solution(args.solution)

    train_x = np.loadtxt(os.path.join(args.data_dir, "train_x.csv"), delimiter=",", skiprows=1)
    train_y = np.loadtxt(os.path.join(args.data_dir, "train_y.csv"), delimiter=",", skiprows=1)
    coords = train_x[:, :2]
    flags = train_x[:, 2] > 0

    perm = np.random.default_rng(args.seed).permutation(len(train_y))
    n_val = int(round(args.val_frac * len(train_y)))
    val, tr = perm[:n_val], perm[n_val:]

    print(f"{args.solution}: {len(tr)} train / {len(val)} validation points")
    t0 = time.time()
    model = sol.Model()
    model.fit_model(coords[tr], train_y[tr], flags[tr])
    t_fit = time.time() - t0

    t0 = time.time()
    predictions, mean, std = model.predict_pollution(coords[val], flags[val])
    t_pred = time.time() - t0

    cost = sol.calculate_cost(train_y[val], np.asarray(predictions, dtype=float), flags[val])
    rmse = float(np.sqrt(np.mean((np.asarray(mean) - train_y[val]) ** 2)))
    res = flags[val]
    under = float(np.mean(np.asarray(predictions)[res] < train_y[val][res])) if res.any() else float("nan")

    print(f"  validation cost          : {cost:.3f}")
    print(f"  RMSE of posterior mean   : {rmse:.3f}")
    print(f"  underpredicted (resid.)  : {100 * under:.1f} %   (~6.9 % expected on noise-free targets; higher here since validation targets are noisy)")
    print(f"  fit time / predict time  : {t_fit:.1f} s / {t_pred:.1f} s")


if __name__ == "__main__":
    sys.exit(main())
