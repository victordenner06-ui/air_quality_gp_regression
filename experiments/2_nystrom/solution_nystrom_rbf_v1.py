import os
import typing
from typing import Tuple
import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigh
from scipy.optimize import brentq
from scipy.stats import norm
from sklearn.cluster import KMeans
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
import matplotlib.pyplot as plt
from matplotlib import cm

# Set `EXTENDED_EVALUATION` to `True` in order to visualize your predictions.
EXTENDED_EVALUATION = True
EVALUATION_GRID_POINTS = 300  # Number of grid points used in extended evaluation

# Cost function constants
COST_W_UNDERPREDICT = 50.0
COST_W_NORMAL = 1.0

class NystroemGP:
    """GP regression with a low-rank Nyström kernel approximation."""

    def __init__(self, length_scale: float, sigma_f: float, sigma_n: float):
        """
        :param length_scale: RBF length-scale l (spatial correlation range)
        :param sigma_f: signal standard deviation (amplitude of the field)
        :param sigma_n: measurement-noise standard deviation
        """
        self.length_scale = length_scale
        self.sigma_f = sigma_f
        self.sigma_n = sigma_n
        self.landmarks = None
        self.W_half_inv = None
        self.chol = None
        self.w_mean = None
        self.y_mean = 0.0

    def _kernel_matrix(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Full RBF kernel matrix k(X1, X2), including the signal amplitude."""
        # Squared Euclidean distances via ||a||^2 - 2 a.b + ||b||^2 (fast, no broadcast)
        X1_sq = np.sum(X1 ** 2, axis=1)[:, None]
        X2_sq = np.sum(X2 ** 2, axis=1)[None, :]
        d2 = X1_sq - 2.0 * (X1 @ X2.T) + X2_sq
        d2 = np.maximum(d2, 0.0)  # guard against tiny negative values
        return (self.sigma_f ** 2) * np.exp(-d2 / (2.0 * self.length_scale ** 2))

    def fit(self, X: np.ndarray, y: np.ndarray, landmarks: np.ndarray) -> None:
        """Fit on ALL n training points (no subsampling needed any more)."""
        self.landmarks = np.asarray(landmarks, dtype=float)
        self.y_mean = float(np.mean(y))
        y_c = y - self.y_mean

        # --- Nyström ingredients ---
        W = self._kernel_matrix(self.landmarks, self.landmarks)   # (m, m)
        W = 0.5 * (W + W.T)                                      # enforce symmetry
        # W^{-1/2} via eigendecomposition, with clipping for numerical safety
        evals, evecs = eigh(W)
        evals = np.clip(evals, 1e-10 * max(evals[-1], 1.0), None)
        self.W_half_inv = (evecs / np.sqrt(evals)) @ evecs.T      # (m, m)

        C = self._kernel_matrix(X, self.landmarks)               # (n, m)
        Phi = C @ self.W_half_inv                                 # features (n, m)

        # --- Bayesian linear regression in feature space ---
        # Posterior of w ~ N(0, I) under y = Phi w + eps, eps ~ N(0, sigma_n^2 I):
        #   precision: B = Phi^T Phi + sigma_n^2 I   (m x m -> cheap Cholesky!)
        #   mean:      w = B^{-1} Phi^T (y - y_mean)
        m = Phi.shape[1]
        B = Phi.T @ Phi + (self.sigma_n ** 2) * np.eye(m)
        c, low = cho_factor(B)
        self.chol = (c, low)
        self.w_mean = cho_solve(self.chol, Phi.T @ y_c)

    def predict(self, Xs: np.ndarray, chunk_size: int = 8192) -> Tuple[np.ndarray, np.ndarray]:
        """
        Posterior mean and std of the LATENT (noise-free) function at Xs.
        Chunked to bound memory for large evaluation grids.
        """
        n = Xs.shape[0]
        means = np.empty(n, dtype=float)
        stds = np.empty(n, dtype=float)
        for start in range(0, n, chunk_size):
            Xb = Xs[start:start + chunk_size]
            Cb = self._kernel_matrix(Xb, self.landmarks)          # (b, m)
            Phib = Cb @ self.W_half_inv                          # (b, m)

            # Posterior mean: E[f*] = phi(x*)^T w_mean + y_mean
            means[start:start + Xb.shape[0]] = Phib @ self.w_mean + self.y_mean

            # Posterior variance: Var[f*] = phi(x*)^T B^{-1} phi(x*), row-wise
            sol = cho_solve(self.chol, Phib.T)                   # B^{-1} Phi_b^T (m, b)
            var = np.einsum('ij,ji->i', Phib, sol)            # diagonal only
            stds[start:start + Xb.shape[0]] = np.sqrt(np.maximum(var, 0.0))
        return means, stds


class Model(object):
    """
    Model for this task.
    You need to implement the fit_model and predict_pollution methods
    without changing their signatures, but are allowed to create additional methods.
    """

    def __init__(self):
        """
        Initialize your model here.
        We already provide a random number generator for reproducibility.
        """
        self.rng = np.random.default_rng(seed=0)

        # -------- Configuration (tune these with spatial CV, not with the public score!) --------
        self.N_HYPER_SAMPLE = 3000          # subsample size for ML-II hyperparameter learning
        self.N_LANDMARKS = 3000            # number of Nyström landmarks m
        self.RES_LANDMARK_FRACTION = 0.25   # fraction of landmarks reserved for residential areas

        # -------- Challenge 3: optimal overprediction shift under the asymmetric cost --------
        # The Bayes-optimal report under weighted squared loss with posterior N(mu, sigma^2)
        # is a = mu + z* sigma, where z* solves
        #     z [Phi(z) + w (1 - Phi(z))] = (w - 1) phi(z),   w = COST_W_UNDERPREDICT
        # (derived by setting d/da E[loss] = 0; Gaussian partial integrals). For w = 50: z* ~ 1.50
        w = COST_W_UNDERPREDICT
        self.residential_shift = brentq(
            lambda z: z * (norm.cdf(z) + w * (1.0 - norm.cdf(z))) - (w - 1.0) * norm.pdf(z),
            0.0, 5.0,
        )


    def fit_model(self, train_coordinates: np.ndarray, train_pollution_targets: np.ndarray, train_residential_flags: np.ndarray):
        """
        Fit your model on the given training data.
        :param train_coordinates: Training features as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :param train_pollution_targets: Training pollution concentrations as a 1d NumPy float array of shape (NUM_SAMPLES,)
        :param train_residential_flags: Binary variable denoting whether the 2D training point is in the residential area (1) or not (0)
        """
        X = np.asarray(train_coordinates, dtype=float)
        y = np.asarray(train_pollution_targets, dtype=float)
        flags = np.asarray(train_residential_flags, dtype=bool)

        # --- Stage 1: learn hyperparameters (l, sigma_f, sigma_n) by maximizing the
        # marginal likelihood on a cost-aware stratified subsample (sklearn GP) ---
        length_scale, sigma_f, sigma_n = self._learn_hyperparameters(X, y, flags)

        # --- Stage 2: cost-aware landmark placement (k-means; oversample residential areas,
        # where underpredictions cost 50x more) ---
        landmarks = self._select_landmarks(X, flags, self.N_LANDMARKS)

        # --- Stage 3: Nyström GP fit on the FULL dataset ---
        self.gp = NystroemGP(length_scale=length_scale, sigma_f=sigma_f, sigma_n=sigma_n)
        self.gp.fit(X, y, landmarks)

        print(f"[fit_model] l = {length_scale:.4f}, sigma_f = {sigma_f:.4f}, "
              f"sigma_n = {sigma_n:.4f}, landmarks = {landmarks.shape[0]}")

    def _learn_hyperparameters(self, X: np.ndarray, y: np.ndarray, flags: np.ndarray) -> Tuple[float, float, float]:
        
        
        n = min(self.N_HYPER_SAMPLE, X.shape[0])
        idx = self.rng.choice(X.shape[0], size=n, replace=False)
        # Hypothesis: smooth spatial field (RBF) + white measurement noise
        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3))
            * RBF(length_scale=0.2, length_scale_bounds=(1e-2, 1e2))
            + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-3, 1e2))
        )
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=True)
        gp.fit(X[idx], y[idx])

        # Unpack learned hyperparameters from (Constant * RBF + White) structure
        amp_kernel = gp.kernel_.k1.k1          # ConstantKernel -> sigma_f^2
        rbf_kernel = gp.kernel_.k1.k2          # RBF            -> length_scale
        white_kernel = gp.kernel_.k2          # WhiteKernel    -> sigma_n^2
        sigma_f = float(np.sqrt(amp_kernel.constant_value))
        length_scale = float(rbf_kernel.length_scale)
        sigma_n = float(np.sqrt(white_kernel.noise_level))
        y_scale = float(np.std(y[idx]))
        sigma_f *= y_scale
        sigma_n *= y_scale
        return length_scale, sigma_f, sigma_n

    def _select_landmarks(self, X: np.ndarray, flags: np.ndarray, n_landmarks: int) -> np.ndarray:
        """
        Cost-aware Nyström landmarks: k-means centers over the whole city,
        plus a dedicated budget of centers inside residential areas.
        """
        n_res_budget = int(round(self.RES_LANDMARK_FRACTION * n_landmarks))
        X_res = X[flags]

        centers = []
        n_res = min(n_res_budget, X_res.shape[0])
        if n_res > 0:
            km_res = KMeans(n_clusters=n_res, n_init=10, random_state=0).fit(X_res)
            centers.append(km_res.cluster_centers_)
        n_glob = n_landmarks - n_res
        if n_glob > 0:
            km_glob = KMeans(n_clusters=n_glob, n_init=10, random_state=0).fit(X)
            centers.append(km_glob.cluster_centers_)
        return np.vstack(centers)

    def predict_pollution(self, test_coordinates: np.ndarray, test_residential_flags: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict the pollution concentration for a given set of city_areas.
        :param test_coordinates: city_areas as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :param test_residential_flags: city_area info for every sample in a form of a bool array (NUM_SAMPLES,)
        :return:
            Tuple of three 1d NumPy float arrays, each of shape (NUM_SAMPLES,),
            containing your predictions, the GP posterior mean, and the GP posterior stddev (in that order)
        """
        # Belief: posterior mean and (noise-free) posterior std at every test point
        gp_mean, gp_std = self.gp.predict(np.asarray(test_coordinates, dtype=float))
        gp_std = gp_std.ravel()

        # Decision rule: symmetric-loss optimum (mean) everywhere, except in
        # residential areas, where underprediction costs 50x more -> report the
        # Bayes-optimal shifted value mu + z* sigma (z* ~ 1.5 for w = 50).
        predictions = gp_mean.copy()
        res = np.asarray(test_residential_flags, dtype=bool)
        predictions[res] += self.residential_shift * gp_std[res]

        return predictions, gp_mean, gp_std


# You don't have to change this function
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


# You don't have to change this function
def is_inside_circle(coordinate, circle_parameters):
    """
    Checks if a coordinate is inside a circle.
    :param coordinate: 2D coordinate
    :param circle_parameters: 3D coordinate of the circle center and its radius
    :return: True if the coordinate is inside the circle, False otherwise
    """
    return (coordinate[0] - circle_parameters[0])**2 + (coordinate[1] - circle_parameters[1])**2 < circle_parameters[2]**2


# You don't have to change this function
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
                    [0.41426299, 0.0641475,  0.04442035],
                    [0.09394051, 0.5759465,  0.08729856],
                    [0.84640867, 0.69947928, 0.04568374],
                    [0.23789282, 0.934214,   0.04039037],
                    [0.82076712, 0.90884372, 0.07434012],
                    [0.09961493, 0.94530153, 0.04755969],
                    [0.88172021, 0.2724369,  0.04483477],
                    [0.9425836,  0.6339977,  0.04979664]])

    residential_flags = np.zeros((grid_coordinates.shape[0],))

    for i,coordinate in enumerate(grid_coordinates):
        residential_flags[i] = any([is_inside_circle(coordinate, circ) for circ in circles])

    return residential_flags


# Don't change the name or the signature of this function
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


# Don't change the name or the signature of this function
def get_city_area_data(train_x: np.ndarray, test_x: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts the city_area information from the training and test features.
    :param train_x: Training features
    :param test_x: Test features
    :return: Tuple of (training features' 2D coordinates, training features' city_area information,
        test features' 2D coordinates, test features' city_area information)
    """
    train_coordinates = train_x[:, :2]
    train_residential_flags = train_x[:, 2] > 0
    test_coordinates = test_x[:, :2]
    test_residential_flags = test_x[:, 2] > 0

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