import os
import sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from uf3.util import user_config
from uf3.util import json_io
from uf3.representation import knots
from uf3.representation import process
from uf3.regression import least_squares
from uf3.data.io import DataCoordinator
from uf3.data.composition import ChemicalSystem
from uf3.representation.bspline import BSplineConfig
from uf3.representation.process import BasisProcessor
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
    data_filename = settings['data_filename']  # input
    features_filename = settings['features_filename']  # input
    training_filename = settings['training_filename']  # output

    model_filename = settings['model_filename']  # output
    zero_tail = settings['zero_tail']
    fit_self_energy = settings['fit_self_energy']
    frac_train = settings['frac_train']
    kappa = settings['kappa']
    regularization_params = settings['regularization_params']
    
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
    knots_file = settings.get("knots_file", None)
    knots_map = knots.parse_knots_file(knots_file, chemical_system)
    bspline_config = BSplineConfig(chemical_system,
                                   knots_map=knots_map,
                                   knot_spacing='custom')
    # Load data
    data_coordinator.dataframe_from_trajectory(data_filename, load=True)
    df_data = data_coordinator.consolidate()
    df_features = process.load_feature_db(features_filename)
    representation = BasisProcessor(chemical_system,
                                    bspline_config)
    regularizer = bspline_config.get_regularization_matrix(
        **regularization_params)
    fixed = bspline_config.get_fixed_tuples(values=0,
                                            one_body=(not fit_self_energy),
                                            upper_bounds=zero_tail,
                                            lower_bounds=False)
    model = WeightedLinearModel(regularizer=regularizer,
                                fixed_tuples=fixed)
    client = ProcessPoolExecutor(max_workers=n_jobs)
    # Split into training and testing sets
    n_train = min(int(len(df_data) * frac_train), len(df_data)-1)
    training_indices = np.random.choice(np.arange(len(df_data)),
                                        n_train,
                                        replace=False)
    training_keys = np.take(df_data.index, training_indices)
    filename = os.path.join(output_directory, training_filename)
    with open(filename, 'w') as f:
        training = list(zip(training_indices, training_keys))
        training = sorted(training, key=lambda t: t[0])
        training = [' '.join([str(value) for value in pair])
                    for pair in training]
        training = '\n'.join(training)
        f.write(training)
        print("Saved training indices:", filename)
    df_train = df_features.loc[training_keys]
    x, y, w = representation.get_training_tuples(df_train,
                                                 kappa,
                                                 data_coordinator)
    # Fit
    model.fit(x, y, weights=w)
    
    # Evaluate
    validation_tuples = representation.get_training_tuples(df_train,
                                                           0,
                                                           data_coordinator)
    x_val, y_val, w_cond = validation_tuples
    # slice entries corresponding to energies per atom
    s_e = df_data.loc[training_keys][data_coordinator.size_key]
    x_e = np.divide(x_val[w_cond == 0].T, s_e.values).T
    y_e = y_val[w_cond == 0] / s_e.values
    # slice entries corresponding to forces
    x_f = x_val[w_cond > 0]
    y_f = y_val[w_cond > 0]
    # predict with solution
    p_e = model.predict(x_e)  # energy per atom
    p_f = model.predict(x_f)

    # Compute root-mean-square error
    rmse_e = np.sqrt(np.mean(np.subtract(y_e, p_e)**2))
    rmse_f = np.sqrt(np.mean(np.subtract(y_f, p_f)**2))
    print("Training RMSE (Energy):", rmse_e, "eV/atom")
    print("Training RMSE (Forces):", rmse_f, "eV/angstrom")
    
    # Arrange solution into pair-interaction potentials
    solutions = least_squares.arrange_coefficients(model.coefficients,
                                                   bspline_config)
    knots_map = bspline_config.knots_map

    filename = os.path.join(output_directory, model_filename)
    json_io.dump_interaction_map({"coefficients": solutions,
                                  "knots": knots_map},
                                 filename=filename,
                                 write=True)
    print("Saved model:", filename)
