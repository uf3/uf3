import os
import sys
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from uf3.util import user_config
from uf3.util import json_io
from uf3.util import plotting
from uf3.regression import least_squares
from uf3.representation import process
from uf3.forcefield import lammps
from uf3.data.io import DataCoordinator
from uf3.data.composition import ChemicalSystem
from uf3.representation.bspline import BSplineBasis
from uf3.representation.process import BasisFeaturizer
from uf3.regression.least_squares import WeightedLinearModel


if __name__ == "__main__":
    if len(sys.argv) == 2:
        settings_filename = "settings.yaml"
    else:
        settings_filename = sys.argv[-1]
    settings = user_config.read_config(settings_filename)

    element_list = settings.get("element_list", ())
    degree = settings['degree']

    output_directory = settings['output_directory']
    data_filename = settings['data_filename']
    features_filename = settings['features_filename']
    training_filename = settings['training_filename']  # output
    model_filename = settings['model_filename']

    plot_fit = settings['plot_fit']
    plot_2B = settings['plot_2B']
    core_correction = settings['core_correction']

    verbose = settings['verbose']
    random_seed = settings['random_seed']
    progress_bars = settings['progress_bars']
    n_jobs = settings['n_jobs']

    np.random.seed(random_seed)
    if verbose >= 1:
        print(settings)
    if len(element_list) < 1:
        raise RuntimeError("No elements specified.")
    if degree != 2 and degree != 3:
        raise ValueError("Degree must be 2 or 3.")
    if n_jobs < 1:
        n_jobs = 1

    # Initialize handlers
    data_coordinator = DataCoordinator()
    chemical_system = ChemicalSystem(element_list=element_list,
                                     degree=degree)
    # Load BSpline config
    filename = os.path.join(output_directory, model_filename)
    model_data = json_io.load_interaction_map(filename)
    coefficients = model_data['coefficients']
    knots_map = model_data['knots']
    # Apply core correction (repulsive core)
    if core_correction > 0:
        for interaction in chemical_system.interactions_map[2]:
            coef_vector = coefficients[interaction]
            knot_sequence = knots_map[interaction]
            coefficients[interaction] = least_squares.postprocess_coefficients(
                coef_vector, core_correction)

    # Export LAMMPS table
    filename = "{}.table".format('-'.join(chemical_system.element_list))
    table_list = []
    pair_list = chemical_system.interactions_map[2]
    for pair in pair_list:
        text = lammps.export_tabulated_potential(knots_map[pair],
                                                 coefficients[pair],
                                                 pair,
                                                 grid=1000,
                                                 filename=None)
        table_list.append(text)
    combined_text = "\n\n\n".join(table_list)
    fname = os.path.join(output_directory, filename)
    with open(fname, "w") as f:
        f.write(combined_text)
        if verbose >= 1:
            print("Exported LAMMPS table:", filename)
    # Load data
    data_coordinator.dataframe_from_trajectory(data_filename, load=True)
    df_data = data_coordinator.consolidate()
    df_features = process.load_feature_db(features_filename)
    bspline_config = BSplineBasis(chemical_system,
                                  knots_map=knots_map,
                                  knot_spacing='custom')
    representation = BasisFeaturizer(chemical_system,
                                     bspline_config)
    model = WeightedLinearModel(bspline_config)
    model.load(coefficients)
    with open(os.path.join(output_directory, training_filename), 'r') as f:
        training_set = f.read()
        training_set = training_set.splitlines()
        training_indices = [int(pair[0]) for pair in training_set]
        training_keys = [pair[1] for pair in training_set]
    testing_keys = df_data.index.difference(training_keys)
    testing_indices = np.where(df_data.index == testing_keys)[0]
    df_test = df_features.loc[testing_keys]
    # Evaluate
    testing_tuples = representation.get_training_tuples(df_test,
                                                        0,
                                                        data_coordinator)
    x_test, y_test, w_cond = testing_tuples
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
    rmse_e = np.sqrt(np.mean(np.subtract(y_e, p_e) ** 2))
    rmse_f = np.sqrt(np.mean(np.subtract(y_f, p_f) ** 2))
    print("Testing RMSE (Energy):", rmse_e, "eV/atom")
    print("Testing RMSE (Forces):", rmse_f, "eV/angstrom")
    # Plot fitting results
    if plot_fit:
        ENERGY_UNITS = "eV/atom"
        FORCE_UNITS = "eV/$\mathrm{\AA}$"

        fig, ax = plt.subplots(figsize=(3.5, 3.5), dpi=160, facecolor='white')
        plotting.density_scatter(y_e, p_e, ax=ax, units=ENERGY_UNITS,
                                 text_size=6, label_size=10)
        ax.set_xlabel('Reference ({})'.format(ENERGY_UNITS))
        ax.set_ylabel('Prediction ({})'.format(ENERGY_UNITS))
        ax.set_title("Energy Testing")
        fig.tight_layout(pad=0.01)
        fig.savefig(os.path.join(output_directory, "energy_testing.png"))
        fig, ax = plt.subplots(figsize=(3.5, 3.5), dpi=160, facecolor='white')
        plotting.density_scatter(y_f, p_f, ax=ax, units=ENERGY_UNITS,
                                 text_size=6, label_size=10)
        ax.set_xlabel('Reference ({})'.format(FORCE_UNITS))
        ax.set_ylabel('Prediction ({})'.format(FORCE_UNITS))
        ax.set_title("Force Testing")
        fig.tight_layout(pad=0.01)
        fig.savefig(os.path.join(output_directory, "force_testing.png"))
    # Plot pair potentials
    if plot_2B:
        for i, pair in enumerate(chemical_system.interactions_map[2]):
            pair_string = '-'.join(pair)
            fig, ax = plt.subplots(figsize=(3.5, 3.5),
                                   dpi=160,
                                   facecolor='white')
            r_min = bspline_config.r_min_map[pair]
            r_max = bspline_config.r_max_map[pair]
            knot_sequence = representation.knots_map[pair]
            coef_vector = coefficients[pair]

            y_min = max(np.min(coef_vector), -1)
            y_max = min(np.max(coef_vector), 1)

            plotting.visualize_splines(coef_vector,
                                       knot_sequence,
                                       ax=ax)
            ax.set_ylim(y_min, y_max)
            ax.set_ylabel('B-Spline Value')
            ax.set_xlabel('$\mathrm{r_{ij}~~(\AA)}$')
            ax.set_title(pair_string)
            fig.tight_layout(pad=0.01)
            fig.savefig(os.path.join(output_directory,
                                     "{}_2B.png".format(pair_string)))

