import sys
import os
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from uf3.util import user_config
from uf3.data import io
from uf3.representation import distances
from uf3.data.io import DataCoordinator
from uf3.data.composition import ChemicalSystem


if __name__ == "__main__":
    if len(sys.argv) == 2:
        settings_filename = "settings.yaml"
    else:
        settings_filename = sys.argv[-1]
    settings = user_config.read_config(settings_filename)
    
    element_list = settings["element_list"]       
    degree = settings['degree']
    
    experiment_path = settings['experiment_path']
    output_directory = settings['output_directory']
    filename_pattern = settings['filename_pattern']
    cache_data = settings['cache_data']  # output
    data_filename = settings['data_filename']
    
    energy_key = settings['energy_key']
    force_key = settings['force_key']
    max_samples = settings['max_samples_per_file']
    min_diff = settings['min_diff']
    vasp_pressure = settings['vasp_pressure']

    analyze_pairs = settings['analyze_pair_distribution']
    analyze_fraction = settings['analyze_fraction']
    
    verbose = settings['verbose']
    random_seed = settings['random_seed']
    progress_bars = settings['progress_bars']
    
    np.random.seed(random_seed)
    if verbose >= 1:
        print(settings)

    if len(element_list) < 1:
        raise RuntimeError("No elements specified.")
    if filename_pattern is None and data_filename is None:
        raise ValueError("Please specify filename_pattern or data_filename.")
    if degree != 2 and degree != 3:
        raise ValueError("Degree must be 2 or 3.")
    # Parse data
    data_coordinator = DataCoordinator(energy_key=energy_key,
                                       force_key=force_key)
    data_paths = io.identify_paths(experiment_path=experiment_path,
                                   filename_pattern=filename_pattern)
    io.parse_with_subsampling(data_paths,
                              data_coordinator,
                              max_samples=max_samples,
                              min_diff=min_diff,
                              energy_key=energy_key,
                              vasp_pressure=vasp_pressure,
                              verbose=verbose)
    if cache_data:
        if os.path.isfile(data_filename):
            if verbose >= 1:
                print("Overwriting...")
            os.remove(data_filename)
        io.cache_data(data_coordinator, data_filename, energy_key=energy_key)
        if verbose >= 1:
            print("Cached data:", data_filename)
    df_data = data_coordinator.consolidate()
    if verbose >= 1:
        n_energies = len(df_data)
        n_forces = int(np.sum(df_data["size"]) * 3)
        print("Number of energies:", n_energies)
        print("Number of forces:", n_forces)
    # Pair distance diagnostics
    if analyze_pairs:
        chemical_system = ChemicalSystem(element_list=element_list,
                                         degree=degree)
        atoms_key = data_coordinator.atoms_key  # default "geometry"
        if analyze_fraction < 1:
            n_samples = len(df_data)
            indices = np.arange(n_samples)
            n = int(n_samples * analyze_fraction)
            indices = np.random.choice(indices, n)
            df_slice = df_data[atoms_key].iloc[indices]
        else:
            df_slice = df_data[atoms_key]
        histograms = distances.summarize_distances(df_slice,
                                                   chemical_system,
                                                   progress_bar=progress_bars)

        bar_width = histograms[1][1] - histograms[1][0]
        pairs = chemical_system.interactions_map[2]
        fig, ax = plt.subplots(1, 
                               len(pairs), 
                               figsize=(len(pairs)*2, 2),
                               dpi=150)
        if len(pairs) == 1:
            ax = [ax]
        for i, pair in enumerate(pairs):
            ax[i].bar(histograms[1][:-1],
                      histograms[0][pair],
                      width=bar_width)
            ax[i].set_title(pair)
            ax[i].plot([0, 10], [1, 1], linestyle='--', color='k')
            ax[i].set_xlim(0, 10)
            ax[i].set_ylim(0, np.max(histograms[0][pair]))
            ax[i].set_xlabel("Pair Distance ($\AA$)")
        ax[0].set_ylabel("Pair Frequency")
        fig.tight_layout()
        fig.savefig("pair_histograms.png")
