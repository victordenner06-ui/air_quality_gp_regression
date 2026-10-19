# Air Quality GP Regression

Spatial prediction of PM2.5 pollution over a 2D city map with a scalable **Gaussian Process**, under an **asymmetric cost** that heavily penalizes underprediction in candidate residential areas.

Course project — *Probabilistic Artificial Intelligence* (PAI), ETH Zurich, Task 1.

---

## Task

- **Input:** relative 2D coordinates `(lon, lat)` in `[0, 1]²` and a binary flag `area_id` (1 = candidate residential area).
- **Output:** PM2.5 concentration at each test location.
- **Data:** 15,189 noisy training measurements (`train_x.csv`, `train_y.csv`) and 3,311 test locations (`test_x.csv`). The test targets are **noise-free**.
- **Metric:** mean weighted squared error. Underpredicting inside a residential area is weighted **×50**, every other error ×1.
- **Pass criterion:** beat the public baseline cost of **21.487**.

## Approach

`solution.py` implements a **Nyström low-rank Gaussian Process** and addresses the three challenges of the task.

### 1. Model selection

Composite kernel:

```
k = σ²_RQ · RationalQuadratic(ℓ_RQ, α) + σ²_M · Matérn_{ν=5/2}(ℓ_M) + σ²_n · White
```

- **Rational Quadratic** — a scale mixture of RBF kernels: models the smooth, large-scale trend at several length-scales.
- **Matérn 5/2** — only twice differentiable: captures sharper local structure (e.g. roads).
- **White noise** — measurement noise of the training data.

Hyperparameters are chosen by **maximizing the marginal likelihood** of an exact `sklearn` GP on a random subsample of 1,000 points (inputs and targets standardized).

### 2. Large-scale learning

An exact GP costs O(n³). We use a **rank-m Nyström approximation** with **m = 5,000 inducing points** chosen by k-means over all training coordinates:

```
k(x, x') ≈ k(x, Z) K_ZZ⁻¹ k(Z, x') = φ(x)ᵀ φ(x'),    φ(x) = L⁻¹ k(Z, x),   K_ZZ = L Lᵀ
```

The GP then becomes **exact Bayesian linear regression on m features** (`f(x) = φ(x)ᵀ w`, `w ~ N(0, I)`), so training only needs an m × m Cholesky factorization — O(n m²) instead of O(n³) — and **all 15,189 points are used**. `ΦᵀΦ` is accumulated in chunks to keep memory bounded. A DTC-style correction adds back the prior variance not captured by the rank-m feature map.

### 3. Asymmetric cost

Under the ×50 penalty the posterior mean is not optimal. For a Gaussian posterior `N(μ, σ²)`, the Bayes-optimal prediction in residential areas is

```
ŷ = μ + z* · σ,   where   z* (Φ(z*) + 50 (1 − Φ(z*))) = 49 φ(z*)   ⇒   z* ≈ 1.485
```

`z*` is computed exactly at initialization with a root finder (`scipy.optimize.brentq`). Outside residential areas, the posterior mean is used.

Because the **test targets are noise-free**, `σ` is the posterior standard deviation of the latent function `f` — it **excludes** the measurement noise. For the same reason, `z` is *not* tuned on a held-out split: the validation targets are noisy, which would inflate `z` by roughly `sqrt(1 + σ²_n / σ²_f,post)`. Predictions are finally clipped at 0 (PM2.5 cannot be negative).

## Repository structure

```
.
├── README.md
├── solution.py                 ← final model (Nyström GP), run by the checker
├── results_check.byte          ← checker output = submission file
├── extended_evaluation.pdf     ← prediction map of the final model
│
├── experiments/                ← alternative models, by approach (see experiments/README.md)
│   ├── 1_undersampling/
│   ├── 2_nystrom/
│   └── 3_local_gp/
├── submissions/
│   └── 1st_submission/         ← snapshot of the first submitted version
├── scripts/
│   ├── validate.py             ← compare any solution file on a train/validation split
│   └── explore_data.py         ← quick look at the training data
├── docs/
│   └── euler-guide.md          ← how to run on the ETH Euler cluster
│
│   course handout — must stay at the root for the checker:
├── train_x.csv  train_y.csv  test_x.csv  test_y.byte
├── checker_client.py  pytransform/
└── Dockerfile  runner.sh  env.yaml  requirements.txt  .dockerignore
```

## Usage

**Official checker (Docker)** — produces the public score and `results_check.byte`:

```bash
bash runner.sh
```

**Quick local run** (Python 3.8, see `requirements.txt`):

```bash
python solution.py
```

Set `EXTENDED_EVALUATION = True` at the top of `solution.py` to also render a full-map prediction plot (`extended_evaluation.pdf`).

**Compare models locally** on a train/validation split of the training data:

```bash
python scripts/validate.py solution.py
python scripts/validate.py experiments/3_local_gp/solution_local_gp_matern.py
```

> The validation targets contain measurement noise while the test targets do not, so the validation cost is higher than the checker's score. Use it to **compare** models, not to predict the public score.

## Interface

| Method | Purpose |
| --- | --- |
| `Model().fit_model(coordinates, targets, flags)` | Standardizes data, selects hyperparameters, fits the Nyström GP |
| `Model().predict_pollution(coordinates, flags)` | Returns shifted predictions, posterior means and posterior stds |
| `calculate_cost(ground_truth, predictions, flags)` | Weighted cost used for scoring |

## Results

| Version | Public score | Notes |
| --- | --- | --- |
| Public baseline | 21.487 | reference to beat |
| `solution.py` (Nyström RQ + Matérn 5/2) | _to fill in_ | final submission |

## Academic integrity

This repository contains a solution to a graded ETH Zurich course project. The PAI course rules forbid publishing project solutions online — keep this repository **private** and share it only with team members.
