import os
import sys
# from concurrent.futures import ProcessPoolExecutor
from dask import distributed
import numpy as np
from uf3.representation import bspline
from uf3.representation import process
from uf3.util import user_config
from uf3.util import json_io
from uf3.data.io import DataCoordinator
from uf3.data.composition import ChemicalSystem
from uf3.representation.bspline import BSplineBasis
from uf3.representation.process import BasisFeaturizer


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
    cache_features = settings['cache_features']
    features_filename = settings['features_filename']  # output
    energy_key = settings['energy_key']
    
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
    data_coordinator = DataCoordinator(energy_key=energy_key)
    chemical_system = ChemicalSystem(element_list=element_list,
                                     degree=degree)
    # handle knots
    read_knots = settings["read_knots"]
    write_knots = settings["write_knots"]
    knots_file = settings.get("knots_file", None)
    if knots_file is not None and os.path.isfile(knots_file) and read_knots:
        knots_map = bspline.parse_knots_file(knots_file, chemical_system)
    else:
        knots_map = {}
    if knots_map:  # successful loading of knots_file
        bspline_config = BSplineConfig(chemical_system,
                                       knots_map=knots_map,
                                       knot_spacing='custom')
    else:
        r_min_map = settings.get("r_min_map", None)
        r_max_map = settings.get("r_max_map", None)
        resolution_map = settings.get("resolution_map", None)
        knot_spacing = settings['knot_spacing']
        if r_min_map is not None:
            r_min_map = {user_config.get_element_tuple(key): value
                         for key, value in r_min_map.items()}
        if r_max_map is not None:
            r_max_map = {user_config.get_element_tuple(key): value
                         for key, value in r_max_map.items()}
        if resolution_map is not None:
            resolution_map = {user_config.get_element_tuple(key): value
                              for key, value in resolution_map.items()}
        bspline_config = BSplineBasis(chemical_system,
                                      r_min_map=r_min_map,
                                      r_max_map=r_max_map,
                                      resolution_map=resolution_map,
                                      knot_spacing=knot_spacing)
    # Load data
    data_coordinator.dataframe_from_trajectory(data_filename, load=True)
    df_data = data_coordinator.consolidate()
    
    # Initialize representation, regularizer, and model
    representation = BasisFeaturizer(chemical_system,
                                     bspline_config)
    knots_file = knots_file or "knots.json"
    if write_knots:
        if os.path.isfile(knots_file) and verbose >= 1:
            print("Overwriting...")
        json_io.dump_interaction_map(representation.knots_map,
                                     filename=knots_file,
                                     write=True)
        if verbose >=1:
            print("Wrote knots:", knots_file)
#    client = ProcessPoolExecutor(max_workers=n_jobs)
    
    cluster = distributed.LocalCluster(n_jobs)
    client = distributed.Client(cluster)

    # Compute representations
    if n_jobs == 1:
        n_batches = 1
    else:
        n_batches = 8  # added granularity for more progress bar updates.
    

    df_features = representation.evaluate_parallel(df_data,
                                                   client,
                                                   n_jobs=n_jobs * n_batches,
                                                   progress_bar=progress_bars)
    if cache_features:
        if os.path.isfile(features_filename):
            print("Overwriting...")
            os.remove(features_filename)
        process.save_feature_db(df_features, features_filename)
        print("Cached features:", features_filename)

