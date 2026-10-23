from typing import Dict, Tuple
from copy import deepcopy
import os
import re
import warnings
import yaml
import uf3
import numpy as np
from ase import symbols as ase_symbols

from uf3.data import io
from uf3.data import composition
from uf3.representation import bspline
from uf3.representation import process
from uf3.regression import least_squares


def get_element_tuple(string):
    """
    Args:
        string (str)

    Returns:
        element_tuple (tuple)
    """
    element_tuple = re.compile("[A-Z][a-z]?").findall(string)
    numbers = {el: ase_symbols.symbols2numbers(el) for el in element_tuple}
    element_tuple = tuple(sorted(element_tuple, key=lambda el: numbers[el]))
    return element_tuple


def type_check(value, reference):
    if reference is None:
        return deepcopy(value)
    type_target = type(reference)
    type_user = type(value)
    if type_target == bool:  # boolean
        if isinstance(value, str):
            choices = {'true': True, 'yes': True, 'on': True, '1': True,
                       'false': False, 'no': False, 'off': False, '0': False}
            if value.lower() not in choices:
                raise ValueError(f'Invalid boolean: {value}')
            return choices[value.lower()]
        return bool(value)
    if type_target in [int, float, np.floating]:  # number
        if type_user in [int, float, np.floating, str]:
            return type_target(value)
    elif type_target in [list, tuple]:  # iterable
        if type_user in [list, tuple]:
            return list(value)
    elif type_target == dict:
        return consistency_check(value, reference)
    elif type_target == type_user:  # other
        return value
    elif type_target == type(None):
        return value
    else:
        raise ValueError("Unknown data type in reference")


def consistency_check(settings, reference):
    if not isinstance(settings, dict):
        raise ValueError('Configuration sections must be a mapping.')
    settings = {key: value for key, value in settings.items()
                if key in reference}
    for key in reference:
        if key in settings:
            settings[key] = type_check(settings[key], reference[key])
        else:
            settings[key] = deepcopy(reference[key])
    return settings


def read_config(settings_filename):
    """
    Read default configuration and configuration from file.
        Parsed settings override defaults only if item types match.

    Args:
        settings_filename (str)

    Returns:
        settings (dict)
    """
    default_config = os.path.join(os.path.dirname(uf3.__file__),
                                  "default_options.yaml")
    with open(default_config, "r") as f:
        default_settings = yaml.safe_load(f)
    with open(settings_filename, "r") as f:
        settings = yaml.safe_load(f)
    return consistency_check({} if settings is None else settings, default_settings)


def generate_handlers(settings: Dict) -> Dict:
    """Initialize and return handlers from configuration dictionary."""
    settings = deepcopy(settings)
    handlers = {}
    if "data" in settings:
        data_settings = settings["data"]["keys"]
        try:
            handlers["data"] = io.DataCoordinator.from_config(data_settings)
        except (KeyError, ValueError):
            warnings.warn('Unable to initialize data handler.', RuntimeWarning)
    if settings.get("elements") and "degree" in settings:
        try:
            chemical_system = composition.ChemicalSystem(
                element_list=settings["elements"],
                degree=settings["degree"])
            handlers["chemical_system"] = chemical_system
        except (KeyError, ValueError):
            warnings.warn('Unable to initialize chemical system.', RuntimeWarning)
    if "basis" in settings and "chemical_system" in handlers:
        basis_block = settings["basis"]
        basis_block.setdefault("element_list", handlers["chemical_system"].element_list)
        basis_block.setdefault("degree", handlers["chemical_system"].degree)
        try:
            bspline_config = bspline.BSplineBasis.from_config(basis_block)
            handlers["basis"] = bspline_config
        except (KeyError, ValueError):
            warnings.warn('Unable to initialize basis handler.', RuntimeWarning)
    if "features" in settings:
        if "chemical_system" in handlers and "basis" in handlers:
            try:
                handlers["features"] = process.BasisFeaturizer(
                    handlers["basis"],
                    fit_forces=settings["features"].get("fit_forces", True),
                    prefix=settings["features"].get("column_prefix", "x"),
                )
            except (KeyError, ValueError):
                warnings.warn('Unable to initialize feature handler.', RuntimeWarning)
    if "model" in settings and "basis" in handlers:
        if os.path.isfile(settings["model"]["model_path"]):
            try:
                model = least_squares.WeightedLinearModel(handlers["basis"])
                model.load(filename=settings["model"]["model_path"])
                handlers["model"] = model
            except (KeyError, ValueError):
                warnings.warn('Unable to load saved model.', RuntimeWarning)
    if "learning" in settings and "basis" in handlers:
        try:
            reg_params = settings["learning"]["regularizer"]
            learning_model = least_squares.WeightedLinearModel(
                handlers["basis"], **reg_params)
            handlers["learning"] = learning_model
        except (KeyError, ValueError):
            warnings.warn('Unable to initialize learning model.', RuntimeWarning)
    return handlers
