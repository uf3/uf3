import os
import re
import sys
import fnmatch
from concurrent.futures import ProcessPoolExecutor

import yaml
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import uf3
from uf3.data.io import DataCoordinator
from uf3.data.composition import ChemicalSystem
from uf3.representation.bspline import BSplineConfig
from uf3.representation.process import BasisProcessor
from uf3.regression.least_squares import WeightedLinearModel

from uf3.representation import distances
from uf3.forcefield import lammps
from uf3.util import plotting
from uf3.util import subsample


if __name__ == "__main__":
    settings_filename = sys.argv[1]

    with open(settings_filename, "r") as f:
        settings = yaml.load(f, Loader=yaml.Loader)

    element_list = settings.get("element_list", None)
    degree = settings.get("degree", 2)
    experiment_path = settings.get("experiment_path", ".")
    output_directory = settings.get("output_directory", ".")
    filename_pattern = settings.get("filename_pattern", None)
    data_filename = settings.get("data_filename", None)
    r_min_map = settings.get("r_min_map", None)
    r_max_map = settings.get("r_max_map", None)
    resolution_map = settings.get("resolution_map", None)
    n_jobs = settings.get("n_jobs", 1)
    frac_train = settings.get("frac_train", 0.8)
    kappa = settings.get("kappa", 0.5)
    regularization_params = settings.get("regularization_params", None)
    plot_fit = settings.get("plot_fit", True)
    cache_representations = settings.get("cache_representations", True)
    max_samples = settings.get("max_samples_per_file", None)
    min_diff = float(settings.get("min_diff", None))

    if element_list is None or not isinstance(element_list, list):
        raise ValueError("No elements specified")
    if regularization_params is None:
        regularization_params = dict(ridge_1b=1e-4,  # L2 (Ridge)
                                     ridge_2b=1e-6,
                                     ridge_3b=1e-5,
                                     curve_2b=1e-5,  # Curvature
                                     curve_3b=1e-4)
    else:
        regularization_params = {key: float(value) for key, value
                                 in regularization_params.items()}
    if filename_pattern is None and data_filename is None:
        raise ValueError("Please specify filename_pattern or data_filename.")
    if degree != 2 and degree != 3:
        raise ValueError("Degree must be 2 or 3.")
    if n_jobs < 1:
        n_jobs = 1
    
    if r_min_map is not None:
        r_min_map = {tuple(re.compile("[A-Z][a-z]?").findall(key)): value
                     for key, value in r_min_map.items()}
    if r_max_map is not None:
        r_max_map = {tuple(re.compile("[A-Z][a-z]?").findall(key)): value
                     for key, value in r_max_map.items()}
    if resolution_map is not None:
        resolution_map = {tuple(re.compile("[A-Z][a-z]?").findall(key)): value
                     for key, value in resolution_map.items()}
    # Search for files
    data_paths = []
    if data_filename is not None:
        data_paths.append(data_filename)
    if filename_pattern is not None:    
        for directory, folders, files in os.walk(experiment_path):
            for filename in files:
                 if fnmatch.fnmatch(filename, filename_pattern):
                    path = os.path.join(directory, filename)
                    data_paths.append(path)

    # Initialize data coordinator and chemical system
    data_coordinator = DataCoordinator()
    chemical_system = ChemicalSystem(element_list=element_list,
                                     degree=degree)
    path_prefix = os.path.commonprefix(data_paths)
    for data_path in data_paths:
        prefix = data_path.replace(path_prefix, "")
        prefix = os.path.dirname(prefix).replace("/", "-")
        df = data_coordinator.dataframe_from_trajectory(data_path,
                                                        prefix=prefix,
                                                        load=False)
        energy_list = df['energy'].values
        subsamples = subsample.farthest_point_sampling(energy_list,
                                                       max_samples=max_samples,
                                                       min_diff=min_diff)
        print("{}/{} samples taken from {}.".format(len(subsamples), 
                                                    len(energy_list),
                                                    prefix))
        df = df.iloc[np.sort(subsamples)]
        data_coordinator.load_dataframe(df, prefix=prefix)
        
                                                   
    df_data = data_coordinator.consolidate()
    n_energies = len(df_data)
    n_forces = int(np.sum(df_data["size"]) * 3)
    print("Number of energies:", n_energies)
    print("Number of forces:", n_forces)
    
    # Initialize representation, regularizer, and model
    bspline_config = BSplineConfig(chemical_system,
                                   r_min_map=r_min_map,
                                   r_max_map=r_max_map,
                                   resolution_map=resolution_map)
    representation = BasisProcessor(chemical_system,
                                    bspline_config)
    regularizer = bspline_config.get_regularization_matrix(
        **regularization_params)
    fixed = bspline_config.get_fixed_tuples(values=0,
                                            one_body=False,
                                            upper_bounds=True,
                                            lower_bounds=False)
    model = WeightedLinearModel(regularizer=regularizer,
                                fixed_tuples=fixed)
    client = ProcessPoolExecutor(max_workers=n_jobs)
    # Compute representations
    n_batches = 8  # added granularity for more progress bar updates.
    df_features = representation.evaluate_parallel(df_data,
                                                   data_coordinator,
                                                   client,
                                                   n_jobs=n_jobs * n_batches)
    if cache_representations:
        resolution_str = "-".join([str(bspline_config.resolution_map[pair])
                                   for pair 
                                   in chemical_system.interactions_map[2]])
        ranges_str = "-".join([str(bspline_config.r_min_map[pair]) + "-" +
                               str(bspline_config.r_max_map[pair])
                               for pair 
                               in chemical_system.interactions_map[2]])
        cache_name = "{}_{}_{}_{}.csv".format(n_energies,
                                              n_forces,
                                              resolution_str,
                                              ranges_str)
        df_features.to_csv(os.path.join(output_directory, cache_name))
                                                   
    # Split into training and testing sets
    n_train = min(int(len(df_data) * frac_train), len(df_data)-1)
    training_indices = np.random.choice(np.arange(len(df_data)),
                                        n_train,
                                        replace=False)
    training_keys = np.take(df_data.index, training_indices)
    testing_indices = np.setdiff1d(np.arange(len(df_data), dtype=int),
                                   training_indices)
    testing_keys = np.take(df_data.index, testing_indices)
    df_train = df_features.loc[training_keys]
    df_test = df_features.loc[testing_keys]
    x, y, w = representation.get_training_tuples(df_train,
                                                 kappa,
                                                 data_coordinator)
    # Fit
    model.fit(x, y, weights=w)
    
    # Evaluate
    x_test, y_test, w_cond = representation.get_training_tuples(df_test,
                                                                0,
                                                                data_coordinator)
    # slice entries corresponding to energies per atom
    s_e = df_data.loc[testing_keys][data_coordinator.size_key]
    x_e = np.divide(x_test[w_cond == 0].T, s_e.values).T
    y_e = y_test[w_cond == 0] / s_e.values
    # slice entries corresponding to forces
    x_f = x_test[w_cond > 0]
    y_f = y_test[w_cond > 0]
    # predict with solution
    p_e = model.predict(x_e)  # energy per atom
    p_f = model.predict(x_f)

    # Compute root-mean-square error
    rmse_e = np.sqrt(np.mean(np.subtract(y_e, p_e)**2))
    rmse_f = np.sqrt(np.mean(np.subtract(y_f, p_f)**2))
    print("Energy RMSE:", rmse_e, "eV/atom")
    print("Forces RMSE:", rmse_f, "eV/angstrom")
    
    # Arrange solution into pair-interaction potentials


    split_indices = np.cumsum(bspline_config.partition_sizes)[:-1]
    solutions_list = np.array_split(model.coefficients,
                                    split_indices)
    solutions = {element: value for element, value
                 in zip(chemical_system.element_list, solutions_list[0])}
    for i, pair in enumerate(chemical_system.interactions_map[2]):
        solutions[pair] = solutions_list[i + 1]
    
    # Export tabulated potential(s)
    table_list = []
    pair_list = chemical_system.interactions_map[2]
    for pair in pair_list:
        text = lammps.export_tabulated_potential(representation.knots_map[pair],
                                                 solutions[pair],
                                                 pair,
                                                 filename=None)
        table_list.append(text)
    combined_text = "\n\n\n".join(table_list)
    table_name = "{}.table".format("-".join([str(el) for el in element_list]))
    table_name = os.path.join(output_directory, table_name)
    with open(table_name, "w") as f:
        f.write(combined_text)

    # Plot fitting results
    if plot_fit:
        y_min = -1
        y_max = 2
        ENERGY_UNITS = "eV/atom"
        FORCE_UNITS = "eV/$\mathrm{\AA}$"

        fig = plt.figure(figsize=(7.48, 4), dpi=160, facecolor='white')
        gs = fig.add_gridspec(ncols=3, nrows=2,
                              width_ratios=[1, 1, 1],
                              height_ratios=[0.5, 1])

        for i, pair in enumerate(chemical_system.interactions_map[2]):
            ax = fig.add_subplot(gs[0, i])
            r_min = r_min_map[pair]
            r_max = r_max_map[pair]
            knot_sequence = representation.knots_map[pair]
            coefficients = solutions[pair]
            plotting.visualize_splines(coefficients,
                                       knot_sequence,
                                       ax=ax,
                                       s_min=y_min,
                                       s_max=y_max,
                                       linewidth=1)
            ax.set_ylabel('B-Spline Value')
            ax.set_xlabel('$\mathrm{r_{ij}~~(\AA)}$')
            ax.set_title(pair)
        ax2 = fig.add_subplot(gs[1, 0])
        plotting.density_scatter(y_e, p_e, ax=ax2, units=ENERGY_UNITS, text_size=6, label_size=10)
        ax2.set_xlabel('Reference ({})'.format(ENERGY_UNITS))
        ax2.set_ylabel('Prediction ({})'.format(ENERGY_UNITS))

        ax3 = fig.add_subplot(gs[1, 1])
        plotting.density_scatter(y_f, p_f, ax=ax3, units=FORCE_UNITS,
                                 text_size=6, label_size=10)
        ax3.set_xlabel('Reference ({})'.format(FORCE_UNITS))
        ax3.set_ylabel('Prediction ({})'.format(FORCE_UNITS))

        ax2.set_title('Energy Predictions')
        ax3.set_title('Force Predictions')
        fig.subplots_adjust(left=0.1, right=0.99,
                            bottom=0.15, top=0.94,
                            wspace=0.6, hspace=0.6)
        fig.savefig(os.path.join(output_directory, "fit_results.png"))
