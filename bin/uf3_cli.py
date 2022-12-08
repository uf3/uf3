#!/usr/bin/env python

import argparse
from cmath import log
from fileinput import filename
import os
from pyexpat import features
import sys
import yaml
import logging
import uf3.util.yaml_tools as yt

accepted_file_formats = ['.xyz', '.pkl']

default_settings = {'verbose': 20, 'outputs_path': './outputs', 'elements': 'W', 'degree': 2, 'seed': 0, 'data': {'db_path': 'data.db', 'max_per_file': -1, 'min_diff': 0.0, 'generate_stats': True, 'progress': 'bar', 'vasp_pressure': False, 'sources': {'path': ['./w-14.xyz', './w-14.xyz'], 'pattern': '*'}, 'keys': {'atoms_key': 'geometry', 'energy_key': 'energy', 'force_key': 'force', 'size_key': 'size'}, 'pickle_path': 'data1.pkl'}, 'basis': {'r_min': 0, 'r_max': 5.5, 'resolution': 15, 'fit_offsets': True, 'trailing_trim': 3, 'leading_trim': 0, 'mask_trim': True, 'knot_strategy': 'linear', 'knots_path': 'knots.json', 'load_knots': False, 'dump_knots': False}, 'features': {
    'db_path': 'data.db', 'features_path': 'features.h5', 'n_cores': 4, 'parallel': 'python', 'fit_forces': True, 'column_prefix': 'x', 'batch_size': 100, 'table_template': 'features_{}'}, 'model': {'model_path': 'model.json'}, 'learning': {'features_path': 'features.h5', 'splits_path': 'splits.json', 'weight': 0.5, 'model_path': 'model.json', 'batch_size': 2500, 'regularizer': {'ridge_1b': 1e-08, 'curvature_1b': 0, 'ridge_2b': 0, 'curvature_2b': 1e-08, 'ridge_3b': 1e-05, 'curvature_3b': 1e-08}}}

# Find all files in the current directory


def find_files():
    files = []
    for file in os.listdir(os.getcwd()):
        if os.path.isfile(file) and os.path.splitext(file)[-1] in accepted_file_formats:
            files.append(file)
    return files


def main(args):

    # Argparse Setup
    parser = argparse.ArgumentParser(prog='uf3',
                                     usage='%(prog)s',
                                     description='Utilities for training and testing UF3 models.')
    subparsers = parser.add_subparsers()

    # Generate starting configuration

    parser_featurize = subparsers.add_parser(
        'config', help='generate a starting config')
    parser_featurize.set_defaults(func=yt.config)
    parser_featurize.add_argument(
        'degree', type=int, help='degree of n-body terms (2 or 3)', default=False)
    parser_featurize.add_argument(
        'atoms', type=str, help='list of atoms', nargs='+', default=False)

    # Collect Data

    parser_collect = subparsers.add_parser(
        'collect', help='collect data and export to pickled file')
    parser_collect.set_defaults(func=yt.collect)
    group = parser_collect.add_mutually_exclusive_group()
    group.add_argument('settings', type=str,
                       help='settings .yaml file', nargs='?', default=False)
    group.add_argument('-f', '--file', type=str,
                       help='single data file to pickle to \'data.pkl\'', default=False)

    # Generate features

    parser_featurize = subparsers.add_parser(
        'featurize', help='compute feature vectors')
    parser_featurize.set_defaults(func=yt.featurize)
    parser_featurize.add_argument(
        'settings', type=str, help='settings .yaml file', default=False)

    # Fit coefficients

    parser_fit = subparsers.add_parser('fit', help='fit model to data')
    parser_fit.set_defaults(func=yt.fit)
    parser_fit.add_argument('settings', type=str,
                            help='settings .yaml file', default=False)

    args = parser.parse_args()

    if not hasattr(args, 'func'):
        parser.print_help()
        sys.exit(1)

    args.func(vars(args))


if __name__ == "__main__":
    main(sys.argv[1:])
