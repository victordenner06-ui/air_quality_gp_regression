import os
import typing
from sklearn.gaussian_process.kernels import *
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.cluster import KMeans
from scipy.linalg import cho_solve, solve_triangular
from scipy.optimize import brentq
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib import cm

EXTENDED_EVALUATION = True
EVALUATION_GRID_POINTS = 300  # Number of grid points used in extended evaluation
# Cost function constants
COST_W_UNDERPREDICT = 50.0
COST_W_NORMAL = 1.0
class Model(object):
    """
    Model for this task: a Nyström low-rank Gaussian Process.

    Strategy:
      1. Model selection: a composite kernel (RBF for the smooth large-scale
         trend + Matérn 5/2 for rougher local structure + white measurement
         noise) has its hyperparameters chosen by maximizing the GP marginal
         likelihood with an exact sklearn GP on a manageable subsample.
      2. Large-scale learning: the full model is a rank-m Nyström approximation
         of the GP. With inducing points Z (k-means over ALL training
         coordinates) the kernel is approximated as
             k(x, x') ~ k(x, Z) K_mm^{-1} k(Z, x') = phi(x)^T phi(x'),
         with the explicit feature map phi(x) = L_mm^{-1} k(Z, x),
         where K_mm = L_mm L_mm^T is the Cholesky factorization.
         The GP is then exactly Bayesian linear regression on m features:
             f(x) = phi(x)^T w,  w ~ N(0, I_m),
         so posterior inference only needs an m x m Cholesky
         (O(n m^2) training instead of O(n^3)) - no sklearn GP model is needed
         for this part.
      3. Asymmetric cost: underprediction in residential areas is weighted 50x,
         so the posterior mean is NOT the optimal prediction there. For a
         Gaussian posterior the Bayes-optimal point prediction under this
         weighted squared loss is mean + z*std, where z* solves
             z*(W - (W-1) * Phi(z)) = (W-1) * phi(z)   with W = 50,
         i.e. z* ~= 1.4852, computed exactly at init with a root finder.
         Empirical tuning on a noisy holdout is deliberately not used:
         validation targets contain measurement noise while the test targets
         do not, so naive holdout tuning would inflate z by
         ~ sqrt(1 + sigma_n^2 / sigma_latent^2).
    """
#config of the size of inducing points and the parameters train point
    M_INDUCING = 4000          # minimum number m of Nyström inducing points
    HPARAM_SUBSAMPLE = 3000    # subsample size for hyperparameter selection
    HPARAM_RESTARTS = 1        # L-BFGS restarts for the marginal likelihood
    CHUNK = 2000              # chunk size to keep memory bounded

    def __init__(self):
        """
        We initialize our model here.
        """
        self.rng = np.random.default_rng(seed=0)

        # preprocessing statistics
        self._x_mean = None
        self._x_std = None
        self._y_mean = 0.0
        self._y_std = 1.0

        # composite-kernel hyperparameters (set in fit_model via marginal likelihood)
        self.rbf_len = 0.3   # RBF lengthscale: smooth, large-scale trend
        self.rbf_var = 1.0   # RBF variance (amplitude squared)
        self.mat_len = 0.1   # Matérn 5/2 lengthscale: rough local structure
        self.mat_var = 0.5   # Matérn 5/2 variance (amplitude squared)
        self.noise_std = 0.1 # measurement noise std

        # Nyström model state
        self._Z = None          # inducing points (standardized space)
        self._L_mm = None       # Cholesky factor of K_mm
        self._L_lam = None      # Cholesky factor of posterior precision of w
        self._w_mean = None     # posterior mean of the feature weights w

        # asymmetric-cost decision rule: predict mean + z_shift * posterior std
        # in residential areas. Exact Bayes optimum for the weighted squared
        # loss (see _optimal_shift), computed instead of hardcoded.
        self.z_shift = self._optimal_shift(COST_W_UNDERPREDICT)

    def fit_model(self, train_coordinates: np.ndarray, train_pollution_targets: np.ndarray, train_residential_flags: np.ndarray):
        """
        Fit the Nyström GP on the given training data.
        """
        X_raw = np.asarray(train_coordinates, dtype=float)
        y_raw = np.asarray(train_pollution_targets, dtype=float).ravel()
        # train_residential_flags is not needed for fitting: the asymmetric
        # cost only matters at prediction time (via the test-side flags).
        n = X_raw.shape[0]

        # ---- 0) standardize inputs and targets ----
        self._x_mean = X_raw.mean(axis=0)
        self._x_std = X_raw.std(axis=0) + 1e-12
        self._y_mean = float(y_raw.mean())
        self._y_std = float(y_raw.std()) + 1e-12
        X = (X_raw - self._x_mean) / self._x_std
        y = (y_raw - self._y_mean) / self._y_std

        # ---- 1) kernel & hyperparameter selection via marginal likelihood ----
        self._select_hyperparameters(X, y)
        # hyperparameter print (lengthscales converted back to map coordinates)
        scale = float(np.mean(self._x_std))
        print(f"[model] kernel (map units): rbf_len={self.rbf_len * scale:.4f}, "
              f"mat_len={self.mat_len * scale:.4f}, rbf_var={self.rbf_var:.3f}, "
              f"mat_var={self.mat_var:.3f}, noise_std={self.noise_std * self._y_std:.3f}")

        # ---- 2) inducing points from ALL training coordinates (k-means) ----
        # Adaptive m (uncapped): the average inducing spacing (~1/sqrt(m) on
        # the unit map) must stay below the learned Matérn lengthscale,
        # otherwise the rank-m approximation smooths over the fine local
        # structure it should capture. Beware: memory grows quadratically -
        # each m x m matrix (A, K_mm, L_mm, L_lam) costs 8*m^2 bytes.
        mat_len_map = self.mat_len * scale
        m_needed = int(np.ceil((1.2 / max(mat_len_map, 1e-4)) ** 2))
        m = min(max(m_needed, self.M_INDUCING), n)
        print(f"[model] inducing points: m = {m} "
              f"(mat_len_map={mat_len_map:.4f}, spacing~{1.0 / np.sqrt(max(m, 1)):.4f}, "
              f"~{8 * m * m / 1e6:.0f} MB per m x m matrix)")
        Z = self._select_inducing_points(X, m)
        self._fit_nystrom(X, y, Z)

    def predict_pollution(self, test_coordinates: np.ndarray, test_residential_flags: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict the pollution concentration for a given set of city_areas.
        """
        Xs = (np.asarray(test_coordinates, dtype=float) - self._x_mean) / self._x_std
        n_test = Xs.shape[0]

        gp_mean = np.zeros(n_test, dtype=float)
        gp_var = np.zeros(n_test, dtype=float)

        for i in range(0, n_test, self.CHUNK):
            Phi = self._features(Xs[i:i + self.CHUNK])          # (chunk, m)
            # posterior mean of the latent function
            gp_mean[i:i + self.CHUNK] = Phi @ self._w_mean
            # posterior variance: diag(Phi @ Lambda^{-1} @ Phi^T)
            T = cho_solve((self._L_lam, True), Phi.T)           # (m, chunk)
            var = np.einsum("ij,ji->i", Phi, T)
            # DTC-style correction: add the prior variance that the rank-m
            # feature map does not capture (SoR would otherwise underestimate
            # the variance away from the inducing points).
            var += np.maximum(self._prior_variance() - np.sum(Phi ** 2, axis=1), 0.0)
            gp_var[i:i + self.CHUNK] = np.maximum(var, 0.0)

        # un-standardize
        gp_mean = gp_mean * self._y_std + self._y_mean
        gp_std = np.sqrt(gp_var) * self._y_std

        # ---- custom decision rule for the asymmetric cost ----
        # Outside residential areas the squared loss is symmetric -> mean.
        # Inside residential areas, underprediction costs 50x more, so the
        # Bayes-optimal prediction under the weighted squared loss is the
        # posterior mean shifted up by ~z* posterior stds.
        predictions = gp_mean.copy()
        residential = np.asarray(test_residential_flags).astype(bool).ravel()
        predictions[residential] += self.z_shift * gp_std[residential]

        return predictions, gp_mean, gp_std
#part for the nystrom solution:

    def _sq_dists(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Pairwise squared euclidean distances between rows of A and B."""
        a2 = np.sum(A ** 2, axis=1)[:, None]
        b2 = np.sum(B ** 2, axis=1)[None, :]
        d2 = a2 + b2 - 2.0 * (A @ B.T)
        np.maximum(d2, 0.0, out=d2)
        return d2

    def _kernel_cross(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        Cross-kernel matrix k(X1, X2) with the current composite kernel:
            k = rbf_var * RBF(rbf_len) + mat_var * Matern_{nu=2.5}(mat_len)
        The RBF models the smooth large-scale trend of the pollution field,
        the Matérn 5/2 the rougher local structure (a real PM2.5 field is only
        once/twice mean-square differentiable, not infinitely smooth as the
        RBF assumes). The Nyström machinery below is kernel-agnostic: any
        kernel that can be evaluated as a cross-covariance works unchanged.
        """
        d2 = self._sq_dists(X1, X2)
        rbf = self.rbf_var * np.exp(-0.5 * d2 / (self.rbf_len ** 2))
        # Matérn with nu = 2.5:  (1 + s + s^2/3) * exp(-s),  s = sqrt(5) d / l
        s = np.sqrt(5.0) * np.sqrt(d2) / self.mat_len
        matern = self.mat_var * (1.0 + s + s ** 2 / 3.0) * np.exp(-s)
        return rbf + matern

    def _prior_variance(self) -> float:
        """Prior variance k(x, x) of the composite kernel (sum of amplitudes)."""
        return self.rbf_var + self.mat_var

    def _features(self, Xc: np.ndarray) -> np.ndarray:
        """
        Nyström feature map phi(x) = L_mm^{-1} k(Z, x).
        phi(x)^T phi(x') = k(x,Z) K_mm^{-1} k(Z,x')  (rank-m kernel approx.)
        """
        K_c = self._kernel_cross(Xc, self._Z)                    # (chunk, m)
        # phi^T = L_mm^{-1} K_c^T  (triangular solve, one row per point)
        return solve_triangular(self._L_mm, K_c.T, lower=True).T

    def _select_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        We maximize the GP marginal likelihood with an exact sklearn GP on a
        subsample (cheap enough at O(s^3)), then reuse the optimized kernel
        hyperparameters in the scalable full-data Nyström model.
        """
        n = X.shape[0]
        s = min(self.HPARAM_SUBSAMPLE, n)
        idx = self.rng.choice(n, size=s, replace=False)

        kernel = (
            ConstantKernel(1.0, constant_value_bounds=(1e-2, 1e3))
            * RBF(length_scale=0.3, length_scale_bounds=(1e-2, 1e1))
            + ConstantKernel(0.5, constant_value_bounds=(1e-2, 1e3))
            * Matern(length_scale=0.1, nu=2.5, length_scale_bounds=(1e-2, 1e1))
            + WhiteKernel(noise_level=0.05, noise_level_bounds=(1e-5, 1e1))
        )
        gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=self.HPARAM_RESTARTS,
            normalize_y=False,
            random_state=0,
        )
        gp.fit(X[idx], y[idx])

        try:
            out = {}
            self._collect_kernel_params(gp.kernel_, out)
            self.rbf_var = out["rbf_var"]
            self.rbf_len = out["rbf_len"]
            self.mat_var = out["mat_var"]
            self.mat_len = out["mat_len"]
            self.noise_std = float(np.sqrt(out["noise"]))
        except (KeyError, AttributeError, TypeError):
            # robust fallback: data-driven defaults
            var = float(np.var(y))
            self.rbf_var, self.rbf_len = 0.7 * var, 0.3
            self.mat_var, self.mat_len = 0.3 * var, 0.1
            self.noise_std = 0.1 * float(np.sqrt(var))
        # guard against degenerate values
        self.rbf_var = max(float(self.rbf_var), 1e-6)
        self.mat_var = max(float(self.mat_var), 1e-6)
        self.rbf_len = max(float(self.rbf_len), 1e-3)
        self.mat_len = max(float(self.mat_len), 1e-3)
        self.noise_std = max(self.noise_std, 1e-4)

    @staticmethod
    def _collect_kernel_params(node, out: dict) -> None:
        """
        Walk the fitted sklearn kernel tree and collect the hyperparameters,
        independent of how Sum/Product nest the components (this is more
        robust than hardcoded get_params() keys, whose prefixes depend on
        the nesting order and the sklearn version).
        """
        if isinstance(node, Sum):
            Model._collect_kernel_params(node.k1, out)
            Model._collect_kernel_params(node.k2, out)
        elif isinstance(node, Product):
            amp, base = None, None
            for child in (node.k1, node.k2):
                if isinstance(child, ConstantKernel):
                    amp = float(child.constant_value)
                else:
                    base = child
            ls = float(np.atleast_1d(base.length_scale).mean())
            if isinstance(base, RBF):
                out["rbf_var"], out["rbf_len"] = amp, ls
            elif isinstance(base, Matern):
                out["mat_var"], out["mat_len"] = amp, ls
        elif isinstance(node, WhiteKernel):
            out["noise"] = float(node.noise_level)

    def _select_inducing_points(self, X: np.ndarray, m: int) -> np.ndarray:
        """
        Inducing points Z via k-means over all training coordinates.
        k-means centroids cover the input space more evenly than a uniform
        random subset at equal m, which improves the Nyström approximation.
        """
        n = X.shape[0]
        if m >= n:
            return X.copy()
        # for very large n, run k-means on a large subsample (still covers space)
        sample = X if n <= 200_000 else X[self.rng.choice(n, size=200_000, replace=False)]
        try:
            km = KMeans(n_clusters=m, n_init=3, max_iter=100, random_state=0)
            km.fit(sample)
            return km.cluster_centers_
        except Exception:
            idx = self.rng.choice(n, size=m, replace=False)
            return X[idx].copy()

    def _fit_nystrom(self, X: np.ndarray, y: np.ndarray, Z: np.ndarray) -> None:
        """
         We fit the rank-m Nyström GP via Bayesian linear regression on the
        feature map phi(x):  f(x) = phi(x)^T w,  w ~ N(0, I_m).

        Posterior over the weights:
            Lambda = phi^T phi / noise_std^2 + I_m
            w | y  ~ N(Lambda^{-1} phi^T y / noise_std^2,  Lambda^{-1})
        Only an m x m Cholesky is needed; all data is used (chunked).
        """
        n = X.shape[0]
        m = Z.shape[0]

        # K_mm and its Cholesky factor (with a small jitter for stability)
        K_mm = self._kernel_cross(Z, Z)
        K_mm[np.diag_indices(m)] += 1e-6 * float(np.mean(np.diag(K_mm)))
        self._L_mm = np.linalg.cholesky(K_mm)
        self._Z = Z

        # accumulate A = Phi^T Phi and b = Phi^T y chunk-wise (memory O(chunk*m))
        A = np.zeros((m, m))
        b = np.zeros(m)
        for i in range(0, n, self.CHUNK):
            Phi = self._features(X[i:i + self.CHUNK])
            A += Phi.T @ Phi
            b += Phi.T @ y[i:i + self.CHUNK]
        A = 0.5 * (A + A.T)  # symmetrize against floating-point round-off

        # posterior precision of w and its Cholesky factor
        Lam = A / (self.noise_std ** 2) + np.eye(m)
        self._L_lam = np.linalg.cholesky(Lam)
        self._w_mean = cho_solve((self._L_lam, True), b / (self.noise_std ** 2))

    @staticmethod
    def _optimal_shift(weight: float) -> float:
        """
        Exact Bayes-optimal shift under the weighted squared loss
            (f - a)^2 * (1 + (W-1) * 1{f > a}),  a = mu + z * sigma,
        for a Gaussian posterior: minimize E[(z - t)^2 (1 + (W-1) 1{t > z})]
        with t ~ N(0, 1), whose first-order condition is
            z * (Phi(z) + W * (1 - Phi(z))) = (W - 1) * phi(z).
        For W = 50 the root is z* ~= 1.4852.
        """
        try:
            return float(brentq(
                lambda z: z * (norm.cdf(z) + weight * (1.0 - norm.cdf(z)))
                          - (weight - 1.0) * norm.pdf(z),
                0.0, 5.0, xtol=1e-10,
            ))
        except Exception:
            return 1.4852  # analytic value for W = 50

def calculate_cost(ground_truth: np.ndarray, predictions: np.ndarray, residential_flags: np.ndarray) -> float:
    """
    Calculates the cost of a set of predictions.

    :param ground_truth: Ground truth pollution levels as a 1d NumPy float array
    :param predictions: Predicted pollution levels as a 1d NumPy float array
    :param residential_flags: city_area info for every sample in a form of a bool array (NUM_SAMPLES,)
    :return: Total cost of all predictions as a single float
    """
    assert ground_truth.ndim == 1 and predictions.ndim == 1 and ground_truth.shape == predictions.shape

    # Unweighted cost
    cost = (ground_truth - predictions) ** 2
    weights = np.ones_like(cost) * COST_W_NORMAL

    # Case i): underprediction
    mask = (predictions < ground_truth) & [bool(residential_flag) for residential_flag in residential_flags]
    weights[mask] = COST_W_UNDERPREDICT

    # Weigh the cost and return the average
    return np.mean(cost * weights)

def is_inside_circle(coordinate, circle_parameters):
    """
    Checks if a coordinate is inside a circle.
    :param coordinate: 2D coordinate
    :param circle_parameters: 3D coordinate of the circle center and its radius
    :return: True if the coordinate is inside the circle, False otherwise
    """
    return (coordinate[0] - circle_parameters[0])**2 + (coordinate[1] - circle_parameters[1])**2 < circle_parameters[2]**2

def determine_residential_flags(grid_coordinates):
    """
    Determines the city_area index for each coordinate in the visualization grid.
    :param grid_coordinates: 2D coordinates of the visualization grid
    :return: 1D array of city_area indexes
    """
    # Circles coordinates
    circles = np.array([[0.5488135, 0.71518937, 0.17167342],
                    [0.79915856, 0.46147936, 0.1567626 ],
                    [0.26455561, 0.77423369, 0.10298338],
                    [0.6976312,  0.06022547, 0.04015634],
                    [0.31542835, 0.36371077, 0.17985623],
                    [0.15896958, 0.11037514, 0.07244247],
                    [0.82099323, 0.09710128, 0.08136552],
                    [0.41426299, 0.0641475 , 0.04442035],
                    [0.09394051, 0.5759465 , 0.08729856],
                    [0.84640867, 0.69947928, 0.04568374],
                    [0.23789282, 0.934214,   0.04039037],
                    [0.82076712, 0.90884372, 0.07434012],
                    [0.09961493, 0.94530153, 0.04755569],
                    [0.88172021, 0.2724369 , 0.04483477],
                    [0.9425836 , 0.6339977 , 0.04979664]])

    residential_flags = np.zeros((grid_coordinates.shape[0],))

    for i,coordinate in enumerate(grid_coordinates):
        residential_flags[i] = any([is_inside_circle(coordinate, circ) for circ in circles])

    return residential_flags

def perform_extended_model_evaluation(model: Model, output_dir: str = '/results'):
    """
    Visualizes the predictions of a fitted model.
    :param model: Fitted model to be visualized
    :param output_dir: Directory in which the visualizations will be stored
    """
    print('Performing extended evaluation')

    # Visualize on a uniform grid over the entire coordinate system
    grid_lat, grid_lon = np.meshgrid(
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
    )
    visualization_grid = np.stack((grid_lon.flatten(), grid_lat.flatten()), axis=1)
    grid_residential_flags = determine_residential_flags(visualization_grid)

    # Obtain predictions, means, and stddevs over the entire map
    predictions, gp_mean, gp_stddev = model.predict_pollution(visualization_grid, grid_residential_flags)
    predictions = np.reshape(predictions, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))
    gp_mean = np.reshape(gp_mean, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))

    vmin, vmax = 0.0, 65.0

    # Plot the actual predictions
    fig, ax = plt.subplots()
    ax.set_title('Extended visualization of task 1')
    im = ax.imshow(predictions, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax)

    # Save figure to pdf
    figure_path = os.path.join(output_dir, 'extended_evaluation.pdf')
    fig.savefig(figure_path)
    print(f'Saved extended evaluation to {figure_path}')

    plt.show()

def get_city_area_data(train_x: np.ndarray, test_x: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts the city_area information from the training and test features.
    :param train_x: Training features
    :param test_x: Test features
    :return: Tuple of (training features' 2D coordinates, training features' city_area information,
        test features' 2D coordinates, test features' city_area information)
    """
    train_x = np.asarray(train_x, dtype=float)
    test_x = np.asarray(test_x, dtype=float)

    # columns: lon, lat, area_id (binary residential-area flag)
    train_coordinates = train_x[:,:2]
    train_residential_flags = train_x[:,2] > 0
    test_coordinates = test_x[:,:2]
    test_residential_flags = test_x[:,2] > 0
    assert train_coordinates.shape[0] == train_residential_flags.shape[0] and test_coordinates.shape[0] == test_residential_flags.shape[0]
    assert train_coordinates.shape[1] == 2 and test_coordinates.shape[1] == 2
    assert train_residential_flags.ndim == 1 and test_residential_flags.ndim == 1

    return train_coordinates, train_residential_flags, test_coordinates, test_residential_flags

# you don't have to change this function
def main():
    # Load the training dataset and test features
    train_x = np.loadtxt('train_x.csv', delimiter=',', skiprows=1)
    train_y = np.loadtxt('train_y.csv', delimiter=',', skiprows=1)
    test_x = np.loadtxt('test_x.csv', delimiter=',', skiprows=1)

    # Extract the city_area information
    train_coordinates, train_residential_flags, test_coordinates, test_residential_flags = get_city_area_data(train_x, test_x)

    # Fit the model
    print('Training model')
    model = Model()
    model.fit_model(train_coordinates, train_y, train_residential_flags)

    # Predict on the test features
    print('Predicting on test features')
    predictions = model.predict_pollution(test_coordinates, test_residential_flags)
    print(predictions)

    if EXTENDED_EVALUATION:
        perform_extended_model_evaluation(model, output_dir='.')


if __name__ == "__main__":
    main()