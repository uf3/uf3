#!/usr/bin/env python

import os
import sys
import yaml
import logging

accepted_file_formats = ['.xyz', '.pkl']

default_settings = {'verbose': 20, 'outputs_path': './outputs', 'elements': 'W', 'degree': 2, 'seed': 0, 'data': {'db_path': 'data.db', 'max_per_file': -1, 'min_diff': 0.0, 'generate_stats': True, 'progress': 'bar', 'vasp_pressure': False, 'sources': {'path': ['./w-14.xyz', './w-14.xyz'], 'pattern': '*'}, 'keys': {'atoms_key': 'geometry', 'energy_key': 'energy', 'force_key': 'force', 'size_key': 'size'}, 'pickle_path': 'data1.pkl'}, 'basis': {'r_min': 0, 'r_max': 5.5, 'resolution': 15, 'fit_offsets': True, 'trailing_trim': 3, 'leading_trim': 0, 'mask_trim': True, 'knot_strategy': 'linear', 'knots_path': 'knots.json', 'load_knots': False, 'dump_knots': False}, 'features': {
    'db_path': 'data.db', 'features_path': 'features.h5', 'n_cores': 4, 'parallel': 'python', 'fit_forces': True, 'column_prefix': 'x', 'batch_size': 100, 'table_template': 'features_{}'}, 'learning': {'features_path': 'features.h5', 'splits_path': 'splits.json', 'weight': 0.5, 'model_path': 'model.json', 'batch_size': 2500, 'regularizer': {'ridge_1b': 1e-08, 'curvature_1b': 0, 'ridge_2b': 0, 'curvature_2b': 1e-08, 'ridge_3b': 1e-05, 'curvature_3b': 1e-08}, "repulsion_correction": False, "min_curvature": 0.0, "repulsion_suffix": "_repulsion"}}

# Find all files in the current directory


def find_files():
    files = []
    for file in os.listdir(os.getcwd()):
        if os.path.isfile(file) and os.path.splitext(file)[-1] in accepted_file_formats:
            files.append(file)
    return files

def get_settings(args):
    if isinstance(args, dict):
        if "settings" in args:
            settings = yaml.load(open(args['settings'], 'r'), Loader=yaml.FullLoader)
        else:
            settings = args
    elif isinstance(args, str):
        settings = yaml.load(open(args, 'r'), Loader=yaml.FullLoader)
    else:
        raise ValueError("Invalid argument. Must be args passed from console, settings dict or path to settings file.")
    return settings


def generate_config_json(degree, atoms):
    """
    Generate config JSON file
    """

    import copy
    settings = copy.deepcopy(default_settings)

    from uf3.data import composition
    chemical_system = composition.ChemicalSystem(element_list=atoms,
                                             degree=degree)

    r_min_map = {}
    r_max_map = {}
    res_map   = {}
    for interaction in ["-".join(i) for i in chemical_system.interactions_map[2]]:
        r_min_map[interaction] = 0
        r_max_map[interaction] = 5
        res_map[interaction]   = 15

    if degree == 3:
        for interaction in ["-".join(i) for i in chemical_system.interactions_map[3]]:
            r_min_map[interaction] = [0,0,0]
            r_max_map[interaction] = [3,3,6]
            res_map[interaction]   = [3,3,6]

    settings['basis']['r_min'] = r_min_map
    settings['basis']['r_max'] = r_max_map
    settings['basis']['resolution'] = res_map
    settings['elements'] = atoms
    settings['degree'] = degree

    return settings


def config(settings):
    """
    Generate a config file
    """

    if settings["atoms"] is None:
        print("Please provide a list of atoms, separated by spaces.")

    if not settings["degree"] in range(2,4):
        print("Valid degrees are 2 and 3.")

    yaml.dump(generate_config_json(settings["degree"], settings["atoms"]), open("input.yaml", 'w'), default_flow_style=None)


def parse_data_files(settings, data_files):
    """
    Parse data files
    """

    from uf3.data import io
    data_coordinator = io.DataCoordinator(atoms_key=settings.get('data', {}).get('keys', {}).get('atoms_key', default_settings['data']['keys']['atoms_key']),
                                          energy_key=settings.get('data', {}).get('keys', {}).get(
                                              'energy_key', default_settings['data']['keys']['energy_key']),
                                          force_key=settings.get('data', {}).get('keys', {}).get(
                                              'force_key', default_settings['data']['keys']['force_key']),
                                          size_key=settings.get('data', {}).get('keys', {}).get('size_key', default_settings['data']['keys']['size_key']))

    for data_file in data_files:
        logging.info('Parsing data file: {}'.format(data_file))

        # Parse 'xyz' files
        if os.path.splitext(data_file)[-1] == '.xyz':
            data_coordinator.dataframe_from_trajectory(data_file,
                                                       prefix='dft')

    # Consolidate data
    df_data = data_coordinator.consolidate()
    return df_data


def get_data_files(settings):
    data_files = []
    if 'data' in settings and 'sources' in settings['data'] and 'path' in settings['data']['sources']:
        path = settings['data']['sources']['path']
        # TODO: Allow for multiple files with wildcards

        # Check if single file
        if isinstance(path, str):
            if os.path.isfile(path) and os.path.splitext(path)[-1] in accepted_file_formats:
                data_files = [path]
            else:
                logging.error(
                    "Specified data file does not exist or is not a supported format. Exiting.")
                sys.exit(1)

        # Check if list of files
        elif isinstance(path, list):
            data_files = []
            for file in path:
                if os.path.isfile(file) and os.path.splitext(file)[-1] in accepted_file_formats:
                    data_files.append(file)
                else:
                    logging.warning(
                        f"Specified data file {file} does not exist or is not a supported format. Skipping.")

        # Throw error if not a string or list
        else:
            logging.error(
                "Specified data files is not a string or list. Exiting.")
            sys.exit(1)

    # Fallback to finding data files in the current directory
    else:
        logging.warning(
            "No data file specified in settings. Searching for data files in current directory.")
        data_files = find_files()

    return data_files


def create_bspline_config(settings, degree, chemical_system):

    # Interactions
    interactions = chemical_system.interactions_map[2].copy()
    if degree == 3:
        interactions += chemical_system.interactions_map[3]
    logging.info(f"Determined interactions: {interactions}")

    disabled_interactions = []

    # Minimum cutoff
    if not 'basis' in settings or not 'r_min' in settings['basis']:
        logging.warning(f"No minimum cutoff specified in settings.")
    r_min = settings.get('basis', {}).get(
        'r_min', default_settings['basis']['r_min'])
    if isinstance(r_min, str):
        r_min = float(r_min)
    elif isinstance(r_min, int):
        r_min = float(r_min)
    if isinstance(r_min, float):
        logging.info(f"Using minimum cutoff of {r_min} for all interactions.")
        r_min_map = {i: r_min if len(i) == 2 else [
            r_min, r_min, r_min * 2] for i in interactions}
    elif isinstance(r_min, dict):

        from uf3.util.json_io import decode_interaction_map
        r_min = decode_interaction_map(r_min)

        for i in interactions:
            if not i in r_min:
                disabled_interactions.append(i)
                # logging.warning(
                #     f"No minimum cutoff specified for interaction {i}. This interaction will not be modeled.")
                r_min[i] = 0 if len(i) == 2 else [0, 0, 0]
        r_min_map = r_min
    else:
        logging.error(
            "Minimum cutoff must be a dict, string, integer, or float. Exiting.")
        sys.exit(1)

    logging.info(f"Using minimum cutoffs: {r_min_map}")

    # Maximum cutoff
    if not 'basis' in settings or not 'r_max' in settings['basis']:
        logging.warning(f"No maximum cutoff specified in settings.")
    r_max = settings.get('basis', {}).get(
        'r_max', default_settings['basis']['r_max'])
    if isinstance(r_max, str):
        r_max = float(r_max)
    elif isinstance(r_max, int):
        r_max = float(r_max)
    if isinstance(r_max, float):
        logging.info(f"Using maximum cutoff of {r_max} for all interactions.")
        r_max_map = {i: r_max if len(i) == 2 else [
            r_max, r_max, r_max * 2] for i in interactions}
    elif isinstance(r_max, dict):

        from uf3.util.json_io import decode_interaction_map
        r_max = decode_interaction_map(r_max)

        for i in interactions:
            if not i in r_max:
                disabled_interactions.append(i)
                # logging.warning(
                #     f"No maximum cutoff specified for interaction {i}. This interaction will not be modeled.")
                r_max[i] = 0 if len(i) == 2 else [0, 0, 0]
        r_max_map = r_max
    else:
        logging.error(
            "Maximum cutoff must be a dict, string, integer, or float. Exiting.")
        sys.exit(1)

    logging.info(f"Using maximum cutoffs: {r_max_map}")

    # Resolution
    if not 'basis' in settings or not 'resolution' in settings['basis']:
        logging.warning(f"No resolution specified in settings.")
    resolution = settings.get('basis', {}).get(
        'resolution', default_settings['basis']['resolution'])
    if isinstance(resolution, str):
        resolution = int(resolution)
    if isinstance(resolution, int):
        logging.info(f"Using resolution of {resolution} for all interactions.")
        resolution_map = {i: resolution if len(
            i) == 2 else [resolution, resolution, resolution * 2] for i in interactions}
    elif isinstance(resolution, dict):

        from uf3.util.json_io import decode_interaction_map
        resolution = decode_interaction_map(resolution)

        for i in interactions:
            if not i in resolution:
                disabled_interactions.append(i)
                # logging.warning(
                #     f"No resolution specified for interaction {i}. This interaction will not be modeled.")
                resolution[i] = 0 if len(i) == 2 else [0, 0, 0]
        resolution_map = resolution
    else:
        logging.error(
            "Resolution must be a dict, string, or integer. Exiting.")
        sys.exit(1)

    if disabled_interactions:
        keys = list(dict.fromkeys(disabled_interactions))
        logging.warning(
                    f"Several interactions were not specified and are not modeled. Specify cutoffs and resolution to include them in the fit: {keys}")
        

    logging.info(f"Using resolutions: {resolution_map}")

    # Basis
    leading_trim = settings.get('basis', {}).get(
        'leading_trim', default_settings['basis']['leading_trim'])
    if isinstance(leading_trim, str):
        try:
            leading_trim = int(leading_trim)
        except:
            logging.error("Leading trim must be an integer. Exiting.")
            sys.exit(1)

    logging.info(f"Using leading trim of {leading_trim}.")

    trailing_trim = settings.get('basis', {}).get(
        'trailing_trim', default_settings['basis']['trailing_trim'])
    if isinstance(trailing_trim, str):
        try:
            trailing_trim = int(trailing_trim)
        except:
            logging.error("Trailing trim must be an integer. Exiting.")
            sys.exit(1)

    logging.info(f"Using trailing trim of {trailing_trim}.")

    offset_1b = settings.get('basis', {}).get(
        'fit_offsets', default_settings['basis']['fit_offsets'])

    from uf3.representation import bspline
    bspline_config = bspline.BSplineBasis(chemical_system,
                                          r_min_map=r_min_map,
                                          r_max_map=r_max_map,
                                          resolution_map=resolution_map,
                                          leading_trim=leading_trim,
                                          trailing_trim=trailing_trim,
                                          offset_1b=offset_1b)

    return bspline_config


def get_chemical_system(element_list, degree):
    from uf3.data import composition
    chemical_system = composition.ChemicalSystem(element_list=element_list,
                                                 degree=degree)

    return chemical_system


def get_degree(settings):
    if not 'degree' in settings:
        logging.warning(
            f"No degree specified in settings. Defaulting to {default_settings['degree']}.")
    degree = settings.get('degree', default_settings['degree'])
    if not isinstance(degree, int):
        try:
            degree = int(degree)
        except:
            logging.error("Degree is not an integer. Exiting.")
            sys.exit(1)
    logging.info(f"Using degree: {degree}")
    return degree


def get_element_list(settings):
    element_list = settings.get('elements', [])
    if isinstance(element_list, str):
        element_list = [element_list]
    if not element_list or not isinstance(element_list, list) or len(element_list) == 0:
        logging.error("No elements specified as list in settings. Exiting.")
        sys.exit(1)

    logging.info("Using elements: {}".format(element_list))
    return element_list


def load_data(settings):
    # Check if pickle exists
    import pandas as pd
    pickle_path = settings.get(
        'data', default_settings['data']).get('pickle_path')
    if os.path.isfile(pickle_path):
        logging.info("Loading data from pickle")
        df_data = pd.read_pickle(pickle_path)

    # Try loading data files
    else:
        logging.info("Loading data from data files")
        data_files = get_data_files(settings)

        # If no data files are found, exit
        if not data_files or len(data_files) == 0:
            logging.error('No data files found. Exiting.')
            sys.exit(1)

        # Parse data files
        df_data = parse_data_files(settings, data_files)
    return df_data


def collect(args):

    # Disable pickled files during collection
    accepted_file_formats.remove('.pkl')

    # If settings have been specified, use them
    if args['settings']:

        # Load settings
        settings = yaml.load(
            open(args['settings'], 'r'), Loader=yaml.FullLoader)

        # Set verbosity
        logging.getLogger().setLevel(settings.get(
            'verbose', default_settings['verbose']))

        # Check if settings specify a data file
        data_files = get_data_files(settings)

        # If no data files are found, exit
        if not data_files or len(data_files) == 0:
            logging.error('No data files found. Exiting.')
            sys.exit(1)

        # Parse data files
        df_data = parse_data_files(settings, data_files)
        df_data.to_pickle(settings.get(
            'data', default_settings['data']).get('pickle_path'))

    # If a single file has been specified, use it
    elif args['file']:
        # Set verbosity
        logging.getLogger().setLevel(default_settings.get('verbose'))

        logging.info("Reading data from file: {}".format(args['file']))

        # Check if file exists and is valid
        if os.path.isfile(args['file']) and os.path.splitext(args['file'])[-1] in accepted_file_formats:

            logging.info("Parsing data file: {}".format(args['file']))

            df_data = parse_data_files({}, [args['file']])
            df_data.to_pickle('data.pkl')

    # If neither have been specified, scan the current directory for data files
    else:
        # Set verbosity
        logging.getLogger().setLevel(default_settings.get('verbose'))

        logging.info(
            "No data file specified. Searching for data files in current directory.")

        data_files = find_files()
        if not data_files or not isinstance(data_files, list) or len(data_files) == 0:
            logging.error('No data files found. Exiting.')
            sys.exit(1)
        else:
            logging.info('Found data files: {}'.format(data_files))
            df_data = parse_data_files({}, data_files)
            df_data.to_pickle('data.pkl')
            logging.info('Data saved to data.pkl')


def featurize(args):
    logging.info("Featurizing data")

    # Load settings
    settings = get_settings(args)

    # Set verbosity
    logging.getLogger().setLevel(settings.get(
        'verbose', default_settings['verbose']))

    # Load data
    df_data = load_data(settings)

    # Elements
    element_list = get_element_list(settings)

    # Degree
    degree = get_degree(settings)

    # Chemical system
    chemical_system = get_chemical_system(element_list, degree)

    # Bspline basis
    bspline_config = create_bspline_config(settings, degree, chemical_system)

    # Get core count
    if not 'features' in settings or not 'n_cores' in settings['features']:
        logging.warning(f"No n_cores specified in settings.")
    n_cores = settings.get('features', {}).get('n_cores', None)
    if n_cores is None:
        import multiprocessing
        n_cores = multiprocessing.cpu_count()

    logging.info(f"Featurizing on {n_cores} cores.")

    # Featurize
    from uf3.representation import process
    representation = process.BasisFeaturizer(bspline_config)

    if settings.get('features', {}).get('parallel', default_settings['features']['parallel']) == 'python':
        from concurrent.futures import ProcessPoolExecutor
        client = ProcessPoolExecutor(max_workers=n_cores)
    elif settings.get('features', {}).get('parallel', default_settings['features']['parallel']) == 'mpi':
        import mpi4py.futures as mpf
        client = mpf.MPIPoolExecutor(max_workers=n_cores)
    else:
        logging.error(
            "Only Python and MPI multiprocessing is supported at this time. Exiting.")
        sys.exit(1)

    filename = get_features_filename(settings)

    batch_size = settings.get('features', {}).get(
        'batch_size', default_settings['features']['batch_size'])
    if isinstance(batch_size, str):
        try:
            batch_size = int(batch_size)
        except:
            logging.error("Batch size must be an integer. Exiting.")
            sys.exit(1)

    logging.info(f"Using batch size of {batch_size}.")

    table_template = settings.get('features', {}).get(
        'table_template', default_settings['features']['table_template'])
    if not isinstance(table_template, str):
        logging.error("Table template must be a string. Exiting.")
        sys.exit(1)

    logging.info(f"Using table template {table_template}.")

    logging.info("Starting featurization...")

    progress = settings.get('data', {}).get(
        'progress', default_settings['data']['progress'])
    representation.batched_to_hdf(filename,
                                  df_data,
                                  client,
                                  n_jobs=n_cores,
                                  batch_size=batch_size,
                                  progress=progress,
                                  table_template=table_template)

    logging.info(f"Featurization complete and saved to {filename}.")


def get_features_filename(settings):
    filename = settings.get('features', {}).get(
        'features_path', default_settings['features']['features_path'])
    if not isinstance(filename, str):
        logging.error("Filename must be a string. Exiting.")
        sys.exit(1)

    logging.info(f"Features stored at {filename}.")
    return filename


def fit(args):

    logging.info("Fitting model")

    # Load settings
    settings = get_settings(args)

    # Set verbosity
    logging.getLogger().setLevel(settings.get(
        'verbose', default_settings['verbose']))

    # Load data
    import pandas as pd
    if "data" in settings:
        if "train_pickle" in settings["data"]:
            pickle_path = settings["data"]["train_pickle"]
        else:
            pickle_path = settings.get(
                'data', default_settings['data']).get('pickle_path')
    if os.path.isfile(pickle_path):
        logging.info(f"Loading data from pickle {pickle_path}")
        df_data = pd.read_pickle(pickle_path)

    # Elements
    element_list = get_element_list(settings)

    # Degree
    degree = get_degree(settings)

    # Chemical system
    chemical_system = get_chemical_system(element_list, degree)

    # Bspline basis
    bspline_config = create_bspline_config(settings, degree, chemical_system)

    ridge_1b = float(settings.get('learning', {}).get('regularizer', {}).get(
        'ridge_1b', default_settings['learning']['regularizer']['ridge_1b']))
    curvature_1b = float(settings.get('learning', {}).get('regularizer', {}).get(
        'curvature_1b', default_settings['learning']['regularizer']['curvature_1b']))
    ridge_2b = float(settings.get('learning', {}).get('regularizer', {}).get(
        'ridge_2b', default_settings['learning']['regularizer']['ridge_2b']))
    curvature_2b = float(settings.get('learning', {}).get('regularizer', {}).get(
        'curvature_2b', default_settings['learning']['regularizer']['curvature_2b']))
    ridge_3b = float(settings.get('learning', {}).get('regularizer', {}).get(
        'ridge_3b', default_settings['learning']['regularizer']['ridge_3b']))
    curvature_3b = float(settings.get('learning', {}).get('regularizer', {}).get(
        'curvature_3b', default_settings['learning']['regularizer']['curvature_3b']))
    weight = float(settings.get('learning', {}).get(
        'weight', default_settings['learning']['weight']))

    logging.info(f"Using ridge_1b = {ridge_1b}.")
    logging.info(f"Using curvature_1b = {curvature_1b}.")
    logging.info(f"Using ridge_2b = {ridge_2b}.")
    logging.info(f"Using curvature_2b = {curvature_2b}.")
    logging.info(f"Using ridge_3b = {ridge_3b}.")
    logging.info(f"Using curvature_3b = {curvature_3b}.")
    logging.info(f"Using weight = {weight}.")

    filename = get_features_filename(settings)
    energy_key = settings.get('data', {}).get('keys', {}).get(
        'energy_key', default_settings['data']['keys']['energy_key'])
    progress = progress = settings.get('data', {}).get(
        'progress', default_settings['data']['progress'])
    batch_size = settings.get('learning', {}).get(
        'batch_size', default_settings['learning']['batch_size'])
    if isinstance(batch_size, str):
        try:
            batch_size = int(batch_size)
        except:
            logging.error("Batch size must be an integer. Exiting.")
            sys.exit(1)

    logging.info(f"Using batch size of {batch_size}.")

    from uf3.regression import least_squares
    regularizer = bspline_config.get_regularization_matrix(
        ridge_1b=ridge_1b, curvature_1b=curvature_1b, ridge_2b=ridge_2b, curvature_2b=curvature_2b, ridge_3b=ridge_3b, curvature_3b=curvature_3b)
    model = least_squares.WeightedLinearModel(
        bspline_config, regularizer=regularizer)

    logging.info("Fitting model...")

    model.fit_from_file(filename,
                        df_data.index,
                        weight=weight,
                        batch_size=2500,
                        energy_key=energy_key,
                        progress=progress)

    model_path = settings.get('learning', {}).get(
        'model_path', default_settings['learning']['model_path'])
    model.to_json(model_path)

    logging.info(f"Model fit complete and stored to {model_path}.")

    if settings.get('learning', {}).get(
        'repulsion_correction', default_settings['learning']['repulsion_correction']):
        from uf3.data import io
        from uf3.data import analyze
        data_coordinator = io.DataCoordinator()
        data_coordinator.load_dataframe(df_data)
        atoms_key = data_coordinator.atoms_key
        analyzer = analyze.DataAnalyzer(model.bspline_config.chemical_system, 
                                        r_cut=10.0,
                                        bins=0.01)
        analyzer.load_entries(df_data[atoms_key])
        analysis = analyzer.analyze()
        min_curvature = settings.get('learning', {}).get(
            'min_curvature', default_settings['learning']['min_curvature'])
        for pair in analysis["lower_bounds"].keys():
            model.fix_repulsion_2b(pair, r_target=analysis["lower_bounds"][pair], min_curvature=min_curvature)

        
        model_repulsion_suffix = settings.get('learning', {}).get(
            'repulsion_suffix', default_settings['learning']['repulsion_suffix'])

        model.to_json(".".join(model_path.split(".")[:-1]) + model_repulsion_suffix + ".json")


def predict(args):

    # Load settings
    settings = get_settings(args)

    # Set verbosity
    logging.getLogger().setLevel(settings.get(
        'verbose', default_settings['verbose']))

    # Load data
    import pandas as pd
    if "data" in settings:
        if "test_pickle" in settings["data"]:
            pickle_path = settings["data"]["test_pickle"]
        else:
            pickle_path = settings.get(
                'data', default_settings['data']).get('pickle_path')
    if os.path.isfile(pickle_path):
        logging.info(f"Loading data from pickle {pickle_path}")
        df_data = pd.read_pickle(pickle_path)
    else:
        logging.error(f"Data not available at {pickle_path}.")

    # Locate features
    filename = get_features_filename(settings)

    # Load model
    from uf3.regression import least_squares
    model_path = settings.get('learning', {}).get(
        'model_path', default_settings['learning']['model_path'])
    model = least_squares.WeightedLinearModel.from_json(model_path)

    logging.info(f"Predicting with model {model_path}")

    # Batched predict
    y_e, p_e, y_f, p_f, rmse_e, rmse_f = model.batched_predict(
        filename, keys=list(df_data.index))

    return y_e, p_e, y_f, p_f, rmse_e, rmse_f

def error(args):

    y_e, p_e, y_f, p_f, rmse_e, rmse_f = predict(args)

    logging.info("Calculating errors")
    
    import numpy as np
    mae_e = np.mean(np.abs(y_e - p_e))  # (eV/atom)
    mae_f = np.mean(np.abs(y_f - p_f))  # (eV/Angstrom)

    # Print results
    logging.info(f"RMSE (eV/atom): {rmse_e:.4f}")
    logging.info(f"RMSE (eV/Angstrom): {rmse_f:.4f}")
    logging.info(f"MAE (eV/atom): {mae_e:.4f}")
    logging.info(f"MAE (eV/Angstrom): {mae_f:.4f}")

    return rmse_e, rmse_f, mae_e, mae_f