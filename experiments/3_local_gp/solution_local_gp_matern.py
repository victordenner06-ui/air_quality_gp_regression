import os
import typing
from sklearn.gaussian_process.kernels import *
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.optimize import brentq
from scipy.stats import norm

# Set `EXTENDED_EVALUATION` to `True` in order to visualize your predictions.
EXTENDED_EVALUATION = True
EVALUATION_GRID_POINTS = 300  # Number of grid points used in extended evaluation

# Cost function constants
COST_W_UNDERPREDICT = 50.0
COST_W_NORMAL = 1.0


class Model(object):

    def __init__(self):
        self.rng = np.random.default_rng(seed=0)
        z = brentq(lambda z: z * (norm.cdf(z) + 50 * (1 - norm.cdf(z))) - 49 * norm.pdf(z), 0, 5)
        self.residential_shift = z

        # Paramètres des GP locaux
        self.n_cells = 5               # la carte est découpée en 5 x 5 régions
        self.nu = 1.5                  # régularité du Matérn (1.5 = variations brutales autorisées)
        self.hp_subsample = 800        # points par zone pour optimiser les hyperparamètres
        self.max_local_points = 3000   # plafond de points pour le GP final de chaque zone
        self.default_margin = 0.08     # marge utilisée avant de connaître le l de la zone

    def _cell_index(self, coords):
        """Numéro (i, j) de la région qui contient chaque point."""
        ij = np.floor(coords * self.n_cells).astype(int)
        ij = np.clip(ij, 0, self.n_cells - 1)
        return ij[:, 0], ij[:, 1]

    def _region_mask(self, X, i, j, margin):
        """Points de X dans la région (i, j) élargie de 'margin' de chaque côté."""
        w = 1.0 / self.n_cells
        return ((X[:, 0] >= i * w - margin) & (X[:, 0] < (i + 1) * w + margin) &
                (X[:, 1] >= j * w - margin) & (X[:, 1] < (j + 1) * w + margin))

    def fit_model(self, train_coordinates: np.ndarray, train_pollution_targets: np.ndarray, train_residential_flags: np.ndarray):
        X = train_coordinates
        y = train_pollution_targets
        self.local_models = {}
        t_start = time.time()

        for i in range(self.n_cells):
            for j in range(self.n_cells):
                t0 = time.time()

                # ---- 1) Données de la zone (avec une marge par défaut) ----
                mask = self._region_mask(X, i, j, self.default_margin)
                Xz, yz = X[mask], y[mask]
                if Xz.shape[0] < 30:                     # zone (quasi) vide : on la saute
                    self.local_models[(i, j)] = None
                    continue

                # Normalisation propre à la zone
                y_mean, y_std = yz.mean(), yz.std() + 1e-12

                # ---- 2) Hyperparamètres de la zone : vraisemblance marginale ----
                n_hp = min(self.hp_subsample, Xz.shape[0])
                idx = self.rng.choice(Xz.shape[0], size=n_hp, replace=False)
                kernel = (ConstantKernel(1.0, (1e-2, 1e2))
                          * Matern(length_scale=0.05, length_scale_bounds=(5e-3, 0.5), nu=self.nu)
                          + WhiteKernel(noise_level=0.05, noise_level_bounds=(1e-4, 1.0)))
                gp_hp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=1,
                                                 normalize_y=False, random_state=0)
                gp_hp.fit(Xz[idx], (yz[idx] - y_mean) / y_std)

                sf2 = gp_hp.kernel_.k1.k1.constant_value     # sigma_f^2 de la zone
                ell = gp_hp.kernel_.k1.k2.length_scale       # l de la zone
                sn2 = gp_hp.kernel_.k2.noise_level           # sigma_n^2 de la zone

                # ---- 3) Marge adaptée au l de la zone, puis GP final ----
                margin = float(np.clip(4 * ell, 0.03, 0.15))
                mask = self._region_mask(X, i, j, margin)
                Xl, yl = X[mask], y[mask]
                if Xl.shape[0] > self.max_local_points:
                    keep = self.rng.choice(Xl.shape[0], size=self.max_local_points, replace=False)
                    Xl, yl = Xl[keep], yl[keep]

                # Noyau "latent" fixé (sans bruit) ; le bruit passe par alpha
                final_kernel = (ConstantKernel(sf2, constant_value_bounds='fixed')
                                * Matern(length_scale=ell, length_scale_bounds='fixed', nu=self.nu))
                gp = GaussianProcessRegressor(kernel=final_kernel, alpha=sn2,
                                              optimizer=None, normalize_y=False)
                gp.fit(Xl, (yl - y_mean) / y_std)

                self.local_models[(i, j)] = {'gp': gp, 'y_mean': y_mean, 'y_std': y_std, 'sf2': sf2}
                print(f'zone ({i},{j}) : {Xl.shape[0]:4d} pts | l={ell:.4f}  sf2={sf2:.3f}  '
                      f'sn2={sn2:.4f}  marge={margin:.3f} | {time.time() - t0:.1f}s')

        print(f'{len(self.local_models)} GP locaux entraînés en {time.time() - t_start:.0f}s')

    def predict_pollution(self, test_coordinates: np.ndarray, test_residential_flags: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = test_coordinates.shape[0]
        gp_mean = np.zeros(n)
        gp_std = np.zeros(n)

        ci, cj = self._cell_index(test_coordinates)
        for (i, j), lm in self.local_models.items():
            sel = (ci == i) & (cj == j)
            if not np.any(sel):
                continue
            if lm is None:                         # zone sans modèle : repli sur la zone voisine la plus proche
                lm = self._nearest_model(i, j)
            m, s = lm['gp'].predict(test_coordinates[sel], return_std=True)
            gp_mean[sel] = m * lm['y_std'] + lm['y_mean']     # retour à l'échelle PM2.5
            gp_std[sel] = s * lm['y_std']

        predictions = gp_mean.copy()
        res = np.asarray(test_residential_flags, dtype=bool)
        predictions[res] += self.residential_shift * gp_std[res]
        return predictions, gp_mean, gp_std

    def _nearest_model(self, i, j):
        """Modèle de la zone non vide la plus proche (sécurité, rarement utilisé)."""
        best, best_d = None, np.inf
        for (a, b), lm in self.local_models.items():
            if lm is not None and (a - i) ** 2 + (b - j) ** 2 < best_d:
                best, best_d = lm, (a - i) ** 2 + (b - j) ** 2
        return best

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

    #TODO: Extract the city_area information from the training and test features
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
