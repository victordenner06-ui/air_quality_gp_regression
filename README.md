# Air Quality GP Regression

Spatial prediction of PM2.5 pollution over a 2D map with a scalable low-rank Gaussian Process, under an asymmetric cost function.

## Task

Given training coordinates with measured pollution levels, predict pollution at test locations. Predictions are scored by a weighted squared error where underpredicting inside residential areas costs 50x more than any other error.

## Approach

`solution.py` implements a **Nyström low-rank Gaussian Process** in three parts:

1. **Model selection** — hyperparameters of a composite kernel (RationalQuadratic for the smooth large-scale trend + Matérn 5/2 for rough local structure + white noise) are chosen by maximizing the GP marginal likelihood with an exact sklearn GP on a 1000-point subsample. The RationalQuadratic is a scale-mixture of RBFs: its `alpha` parameter lets a single component model variation at multiple length-scales.

2. **Scalable fitting** — the full GP is approximated at rank *m* using inducing points from k-means over all training coordinates. With the explicit feature map `phi(x) = L_mm^-1 k(Z, x)`, the GP becomes exact Bayesian linear regression on *m* features, so training needs only an *m x m* Cholesky (O(n m²) instead of O(n³)) and all data is used. The number of inducing points *m* is adapted so their spacing stays below the learned Matérn lengthscale (minimum 5000).

3. **Asymmetric-cost prediction** — under a 50x underprediction penalty, the posterior mean is not optimal. For a Gaussian posterior, the Bayes-optimal point prediction in residential areas is `mean + z* · std`, where `z* ≈ 1.485` solves the first-order condition of the weighted loss (computed at init with a root finder). Outside residential areas, the plain posterior mean is used.

## Usage

```bash
./runner.sh
```

Requires `train_x.csv`, `train_y.csv`, and `test_x.csv` in the working directory. Predictions are printed to stdout.

Set `EXTENDED_EVALUATION = True` at the top of `solution.py` to also render a full-map prediction visualization (`extended_evaluation.pdf`).

## Interface

| Method | Purpose |
| --- | --- |
| `Model().fit_model(coordinates, targets, flags)` | Standardizes data, selects hyperparameters, fits the Nyström GP |
| `Model().predict_pollution(coordinates, flags)` | Returns shifted predictions, posterior means, and posterior stds |
| `calculate_cost(ground_truth, predictions, flags)` | Weighted cost used for scoring |
