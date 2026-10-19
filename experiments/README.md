# Experiments

Alternative models explored during the project, grouped by how they handle the O(n³) cost of exact GPs. The final model is [`../solution.py`](../solution.py).

Every file is a complete drop-in replacement for `solution.py` (same `Model` interface) and uses the same asymmetric-cost rule: `mean + z* · std` in residential areas, `z* ≈ 1.485`.

## 1. Undersampling — `1_undersampling/`

| File | Kernel | Notes |
| --- | --- | --- |
| `solution_undersampling_rbf.py` | Constant × RBF + White | Exact GP on 5,000 random points. First working version; slow (15 optimizer restarts) and discards 2/3 of the data. |

## 2. Nyström low-rank approximation — `2_nystrom/`

| File | Kernel | Inducing points | Notes |
| --- | --- | --- | --- |
| `solution_nystrom_rbf_v1.py` | Constant × RBF + White | 3,000 | First low-rank version. |
| `solution_nystrom_matern_v1.py` | Constant × Matérn + White | 3,000 | Matérn for sharper local variations. |
| `solution_nystrom_rbf_v2.py` | RBF + White | 5,000 (k-means) | Final architecture with a single RBF kernel (ablation). |
| `solution_nystrom_rbf_matern.py` | RBF + Matérn 5/2 + White | 5,000 (k-means) | Direct predecessor of `solution.py` (which replaces the RBF by a Rational Quadratic). |

## 3. Local GPs — `3_local_gp/`

The map is split into 5 × 5 regions; each region gets its own GP trained on its points plus an overlap margin, and each test point is predicted by the GP of its region. All data is used.

| File | Kernel | Hyperparameters | Notes |
| --- | --- | --- | --- |
| `solution_local_gp_rbf.py` | Constant × RBF | global (one fit on 2,500 points) | Margin 3ℓ. Noise passed via `alpha`, so σ excludes measurement noise. |
| `solution_local_gp_matern.py` | Constant × Matérn 3/2 | per zone (fit on 800 points per zone) | Adaptive margin 4ℓ, per-zone normalization. |

## Running a variant

```bash
# compare locally (from the repository root)
python scripts/validate.py experiments/3_local_gp/solution_local_gp_matern.py

# official checker: copy the variant over solution.py, then
bash runner.sh
```
