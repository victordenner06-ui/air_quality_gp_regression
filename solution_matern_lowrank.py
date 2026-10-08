import os
import time
import typing
from sklearn.gaussian_process.kernels import *
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.linalg import cho_factor, cho_solve, eigh
from scipy.optimize import brentq, minimize
from scipy.stats import norm
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import cdist
from scipy.special import kv, gamma


# Set `EXTENDED_EVALUATION` to `True` in order to visualize your predictions.
EXTENDED_EVALUATION = False
EVALUATION_GRID_POINTS = 300  # Number of grid points used in extended evaluation

# Cost function constants
COST_W_UNDERPREDICT = 50.0
COST_W_NORMAL = 1.0

CHUNK = 5000  # rows processed at a time (limits memory)


def _rbf(X1: np.ndarray, X2: np.ndarray, length_scale: float, sigma_f: float) -> np.ndarray:
    """RBF kernel matrix k(X1, X2) = sigma_f^2 exp(-||x - x'||^2 / (2 l^2))."""
    X1_sq = np.sum(X1 ** 2, axis=1)[:, None]
    X2_sq = np.sum(X2 ** 2, axis=1)[None, :]
    d2 = np.maximum(X1_sq - 2.0 * (X1 @ X2.T) + X2_sq, 0.0)
    return (sigma_f ** 2) * np.exp(-d2 / (2.0 * length_scale ** 2))

def _matern(X1, X2, length_scale, sigma_f, nu=1.5):
    """Matérn kernel matrix."""
    d = cdist(X1, X2, metric="euclidean")
    r = np.sqrt(2 * nu) * d / length_scale

    K = np.empty_like(r)

    if nu == 0.5:
        K = np.exp(-r)

    elif nu == 1.5:
        K = (1 + r) * np.exp(-r)

    elif nu == 2.5:
        K = (1 + r + r**2 / 3) * np.exp(-r)

    else:
        # General Matérn
        K = np.ones_like(r)
        mask = r > 0
        K[mask] = (
            2 ** (1 - nu)
            / gamma(nu)
            * r[mask] ** nu
            * kv(nu, r[mask])
        )

    return sigma_f**2 * K


EIG_TOL = 1e-8  # eigenvalues of K_mm below EIG_TOL * max are dropped (this sets the rank r)


def _inv_sqrt_psd(W: np.ndarray) -> np.ndarray:
    """
    Projection P = U_r Lambda_r^{-1/2}, shape (m, r): the pseudo-inverse square root of W.
    Eigenvalues below EIG_TOL * max are dropped, so features are Phi = k(X, Z) @ P with r columns.
    """
    W = 0.5 * (W + W.T)
    evals, evecs = eigh(W)
    keep = evals > EIG_TOL * evals[-1]
    return evecs[:, keep] / np.sqrt(evals[keep])


class NystroemGP:
    """GP regression with a low-rank Nystroem kernel approximation."""

    def __init__(self, length_scale: float, sigma_f: float, sigma_n: float):
        self.length_scale = length_scale
        self.sigma_f = sigma_f
        self.sigma_n = sigma_n
        self.landmarks = None
        self.W_half_inv = None
        self.chol = None
        self.w_mean = None
        self.y_mean = 0.0

    def _kernel_matrix(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        return _matern(
        X1, X2,
        self.length_scale,
        self.sigma_f,
        nu=1.5
        )

    def fit(self, X: np.ndarray, y: np.ndarray, landmarks: np.ndarray) -> None:
        """Fit on ALL n training points."""
        self.landmarks = np.asarray(landmarks, dtype=float)
        self.y_mean = float(np.mean(y))
        y_c = y - self.y_mean

        W = self._kernel_matrix(self.landmarks, self.landmarks)   # (m, m)
        self.W_half_inv = _inv_sqrt_psd(W)                        # (m, m)

        # Bayesian linear regression on the features Phi = k(X, Z) W^{-1/2}.
        # Phi^T Phi and Phi^T y are accumulated chunk by chunk.
        m = self.W_half_inv.shape[1]   # r: number of eigenvalues kept
        print(f"[fit] landmarks m = {self.landmarks.shape[0]}, rank r = {m}")
        B = (self.sigma_n ** 2) * np.eye(m)
        b = np.zeros(m)
        for s in range(0, X.shape[0], CHUNK):
            Phi = self._kernel_matrix(X[s:s + CHUNK], self.landmarks) @ self.W_half_inv
            B += Phi.T @ Phi
            b += Phi.T @ y_c[s:s + CHUNK]
        self.chol = cho_factor(B)
        self.w_mean = cho_solve(self.chol, b)

    def predict(self, Xs: np.ndarray, chunk_size: int = 8192) -> typing.Tuple[np.ndarray, np.ndarray]:
        """Posterior mean and std of the latent (noise-free) function at Xs."""
        n = Xs.shape[0]
        means = np.empty(n, dtype=float)
        stds = np.empty(n, dtype=float)
        for start in range(0, n, chunk_size):
            Xb = Xs[start:start + chunk_size]
            Phib = self._kernel_matrix(Xb, self.landmarks) @ self.W_half_inv   # (b, m)

            means[start:start + Xb.shape[0]] = Phib @ self.w_mean + self.y_mean

            # Posterior covariance of the weights is sigma_n^2 * B^{-1}
            sol = cho_solve(self.chol, Phib.T)
            var = (self.sigma_n ** 2) * np.einsum('ij,ji->i', Phib, sol)
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

        self.gp = None
        self.N_LANDMARKS = 3000             # Nystroem landmarks m for the final fit
        self.N_HYPER_LANDMARKS = 300        # landmarks used by the "lowrank" hyperparameter search
        # How to learn (l, sigma_f, sigma_n): "sklearn_subset" = exact GP likelihood on a random
        # subset (reliable), or "lowrank" = low-rank likelihood on all samples (biased if few landmarks)
        self.HYPER_METHOD = "sklearn_subset"
        self.N_HYPER_SAMPLE = 2000          # subset size for the exact GP
        self.HYPER_RESTARTS = 1             # optimizer restarts for the exact GP
        self.RES_LANDMARK_FRACTION = 0.25   # fraction of landmarks reserved for residential areas

        # Optimal overprediction shift under the weighted squared loss:
        # prediction = mu + z * sigma, where z solves
        #     z [Phi(z) + w (1 - Phi(z))] = (w - 1) phi(z),   w = COST_W_UNDERPREDICT
        w = COST_W_UNDERPREDICT
        self.residential_shift = brentq(
            lambda z: z * (norm.cdf(z) + w * (1.0 - norm.cdf(z))) - (w - 1.0) * norm.pdf(z),
            0.0, 5.0,
        )

    # Don't change the name or the signature of this function
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

        # Stage 1: hyperparameters from the marginal likelihood of the low-rank model (all samples)
        t0 = time.time()
        length_scale, sigma_f, sigma_n = self._learn_hyperparameters(X, y)
        print(f"[timing] hyperparameters: {time.time() - t0:.1f}s")

        # Stage 2: cost-aware landmark placement
        t0 = time.time()
        landmarks = self._select_landmarks(X, flags, self.N_LANDMARKS)
        print(f"[timing] landmarks: {time.time() - t0:.1f}s")

        # Stage 3: Nystroem GP fit on the full dataset
        t0 = time.time()
        self.gp = NystroemGP(length_scale=length_scale, sigma_f=sigma_f, sigma_n=sigma_n)
        self.gp.fit(X, y, landmarks)
        print(f"[timing] final fit: {time.time() - t0:.1f}s")

        print(f"[fit_model] l = {length_scale:.4f}, sigma_f = {sigma_f:.4f}, "
              f"sigma_n = {sigma_n:.4f}, landmarks = {landmarks.shape[0]}")

    def _neg_log_marginal_likelihood(self, log_theta, X, y_c, Z) -> float:
        """
        Negative log marginal likelihood of y ~ N(0, Phi Phi^T + sn^2 I) for all n points.
        With B = Phi^T Phi + sn^2 I and b = Phi^T y (Woodbury + determinant lemma):
            nlml = 0.5 * [ (y^T y - b^T B^{-1} b) / sn^2 + (n - m) log sn^2 + log|B| + n log 2 pi ]
        """
        ell, sf, sn = np.exp(log_theta)
        n, m = X.shape[0], Z.shape[0]

        W_half_inv = _inv_sqrt_psd(_rbf(Z, Z, ell, sf))

        CtC = np.zeros((m, m))
        Cty = np.zeros(m)
        for s in range(0, n, CHUNK):
            C = _rbf(X[s:s + CHUNK], Z, ell, sf)
            CtC += C.T @ C
            Cty += C.T @ y_c[s:s + CHUNK]

        r = W_half_inv.shape[1]
        B = W_half_inv.T @ CtC @ W_half_inv + (sn ** 2) * np.eye(r)
        b = W_half_inv.T @ Cty
        try:
            c, low = cho_factor(B)
        except np.linalg.LinAlgError:
            return 1e30
        quad = (y_c @ y_c - b @ cho_solve((c, low), b)) / (sn ** 2)
        logdet = (n - r) * np.log(sn ** 2) + 2.0 * np.sum(np.log(np.diag(c)))
        return 0.5 * (quad + logdet + n * np.log(2.0 * np.pi))

    def _learn_hyperparameters_exact(self, X: np.ndarray, y: np.ndarray) -> typing.Tuple[float, float, float]:
        """Exact GP marginal likelihood on a random subset; only used to choose l, sigma_f, sigma_n."""
        n = min(self.N_HYPER_SAMPLE, X.shape[0])
        idx = self.rng.choice(X.shape[0], size=n, replace=False)
        kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(
            length_scale=0.1,
            length_scale_bounds=(1e-2, 1e1),
            nu=1.5
        )
        + WhiteKernel(noise_level=0.3, noise_level_bounds=(1e-3, 1e1))
)
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=self.HYPER_RESTARTS,
                                      normalize_y=True, random_state=0)
        gp.fit(X[idx], y[idx])
        y_scale = float(np.std(y[idx]))          # sklearn learned the values in normalised units
        sigma_f = float(np.sqrt(gp.kernel_.k1.k1.constant_value)) * y_scale
        length_scale = float(gp.kernel_.k1.k2.length_scale)
        sigma_n = float(np.sqrt(gp.kernel_.k2.noise_level)) * y_scale
        print(f"[hyper] exact GP on {n} points: {gp.kernel_}")
        return length_scale, sigma_f, sigma_n

    def _learn_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> typing.Tuple[float, float, float]:
        """Learn (l, sigma_f, sigma_n), either from an exact GP on a subset or from the low-rank likelihood."""
        if self.HYPER_METHOD == "sklearn_subset":
            return self._learn_hyperparameters_exact(X, y)
        y_scale = float(np.std(y))
        y_c = (y - np.mean(y)) / y_scale          # normalised targets: amplitudes are O(1)

        km = MiniBatchKMeans(n_clusters=self.N_HYPER_LANDMARKS, n_init=1,
                             batch_size=2048, random_state=0).fit(X)
        Z = km.cluster_centers_

        bounds = [(np.log(1e-2), np.log(1e1)),     # length scale
                  (np.log(1e-2), np.log(1e2)),     # sigma_f (normalised units)
                  (np.log(1e-3), np.log(1e1))]     # sigma_n (normalised units)
        starts = [(0.1, 1.0, 0.3)]   # add more starts only if the result looks unstable

        best = None
        for ell0, sf0, sn0 in starts:
            res = minimize(self._neg_log_marginal_likelihood, np.log([ell0, sf0, sn0]),
                           args=(X, y_c, Z), method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 30})
            if best is None or res.fun < best.fun:
                best = res
        length_scale, sigma_f, sigma_n = np.exp(best.x)
        return float(length_scale), float(sigma_f * y_scale), float(sigma_n * y_scale)

    def _select_landmarks(self, X: np.ndarray, flags: np.ndarray, n_landmarks: int) -> np.ndarray:
        """
        Cost-aware Nystroem landmarks: k-means centers over the whole city,
        plus a dedicated budget of centers inside residential areas.
        """
        n_res_budget = int(round(self.RES_LANDMARK_FRACTION * n_landmarks))
        X_res = X[flags]

        centers = []
        n_res = min(n_res_budget, X_res.shape[0])
        if n_res > 0:
            km_res = MiniBatchKMeans(n_clusters=n_res, n_init=1,
                                     batch_size=2048, random_state=0).fit(X_res)
            centers.append(km_res.cluster_centers_)
        n_glob = n_landmarks - n_res
        if n_glob > 0:
            km_glob = MiniBatchKMeans(n_clusters=n_glob, n_init=1,
                                      batch_size=2048, random_state=0).fit(X)
            centers.append(km_glob.cluster_centers_)
        return np.vstack(centers)

    # Don't change the name or the signature of this function
    def predict_pollution(self, test_coordinates: np.ndarray, test_residential_flags: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict the pollution concentration for a given set of city_areas.
        :param test_coordinates: city_areas as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :param test_residential_flags: city_area info for every sample in a form of a bool array (NUM_SAMPLES,)
        :return:
            Tuple of three 1d NumPy float arrays, each of shape (NUM_SAMPLES,),
            containing your predictions, the GP posterior mean, and the GP posterior stddev (in that order)
        """
        gp_mean, gp_std = self.gp.predict(np.asarray(test_coordinates, dtype=float))
        gp_std = gp_std.ravel()

        # Mean everywhere, except in residential areas where underprediction costs 50x:
        # report mu + z* sigma there.
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

# You do not need to change the name or the signature of this function; convenience helper used by main() 
def get_city_area_data(train_x: np.ndarray, test_x: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts the city_area information from the training and test features.
    :param train_x: Training features
    :param test_x: Test features
    :return: Tuple of (training features' 2D coordinates, training features' city_area information,
        test features' 2D coordinates, test features' city_area information)
    """
    train_coordinates = np.zeros((train_x.shape[0], 2), dtype=float)
    train_residential_flags = np.zeros((train_x.shape[0],), dtype=bool)
    test_coordinates = np.zeros((test_x.shape[0], 2), dtype=float)
    test_residential_flags = np.zeros((test_x.shape[0],), dtype=bool)

    # Columns are: lon, lat, area_id
    train_coordinates = train_x[:, :2].astype(float)
    train_residential_flags = train_x[:, 2] > 0
    test_coordinates = test_x[:, :2].astype(float)
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