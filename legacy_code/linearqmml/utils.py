import re

import numpy as np
try:
    import tqdm
except ImportError:
    pass  # soft dependency


def tqdm_wrap(iterable):
    """tqdm is an optional dependency; wraps progress bar"""
    try:
        tqdm.tqdm._instances.clear()
    except (AttributeError, NameError):
        pass  # no running instances

    try:
        return tqdm.tqdm(iterable)
    except NameError:
        return iterable  # could not import tqdm


def natural_sort(s, _nsre=re.compile('([0-9]+)')):
    """sorting with support for numbers in strings"""
    return [int(text) if text.isdigit() else text.lower()
            for text in _nsre.split(s)]


def get_spherical_scaling(r_range, density):
    """RDF scaling"""
    bin_width = r_range[1] - r_range[0]
    scaling = 4 * np.pi * r_range**2 * density * bin_width
    return scaling


def get_cosine_smoothing(n_bins):
    """Smooth decay from Behler & Parrinello 2007"""
    smoothing =  0.5*(np.cos(np.pi*(np.arange(n_bins)/n_bins))+1)
    return smoothing


def get_morse_potential(r_range, D0=0.3429, A=1.3588, R0=2.6260):
    """Morse potential"""
    morse = (D0 * (np.exp(-2 * A * (r_range - R0))
                   - 2 * np.exp(-A * (r_range - R0)))) / 2
    return morse

# def quick_access(h5_filename, keys, tq=False):
#     """access HDF5 file with keys"""
#     with h5py.File(h5_filename, 'r') as f:
#         if tq == True:
#             keys = tqdm_wrap(keys)
#         data = [f[key][()] for key in keys]
#         return data



