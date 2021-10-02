from typing import Dict, Tuple
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
    type_target = type(reference)
    type_user = type(value)
    if type_target == bool:  # boolean
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
    settings = {key: value for key, value in settings.items()
                if key in reference}
    for key in reference:
        if key in settings:
            settings[key] = type_check(settings[key], reference[key])
        else:
            settings[key] = reference[key]
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
        default_settings = yaml.load(f, Loader=yaml.Loader)
    with open(settings_filename, "r") as f:
        settings = yaml.load(f, Loader=yaml.Loader)

    for key in settings:
        if key not in default_settings:
            continue
        settings[key] = type_check(settings[key], default_settings[key])
    return settings


def generate_handlers(settings: Dict) -> Dict:
    """Initialize and return handlers from configuration dictionary."""
    handlers = {}
    if "data" in settings:
        data_settings = settings["data"]["keys"]
        try:
            handlers["data"] = io.DataCoordinator.from_config(data_settings)
        except (KeyError, ValueError):
            pass
    if "elements" in settings and "degree" in settings:
        try:
            chemical_system = composition.ChemicalSystem(
                element_list=settings["elements"],
                degree=settings["degree"])
            handlers["chemical_system"] = chemical_system
        except (KeyError, ValueError):
            pass
    if "basis" in settings and "chemical_system" in handlers:
        basis_block = settings["basis"]
        basis_block["chemical_system"] = handlers["chemical_system"]
        try:
            bspline_config = bspline.BSplineBasis.from_config(basis_block)
            handlers["basis"] = bspline_config
        except (KeyError, ValueError):
            pass
    if "features" in settings:
        if "chemical_system" in handlers and "basis" in handlers:
            try:
                handlers["features"] = process.BasisFeaturizer(
                    handlers["chemical_system"],
                    handlers["basis"],
                    fit_forces=settings.get("fit_forces", True),
                    prefix=settings.get("feature_prefix", "x"),
                )
            except (KeyError, ValueError):
                pass
    if "model" in settings and "basis" in handlers:
        if os.path.isfile(settings["model"]["model_path"]):
            try:
                model = least_squares.WeightedLinearModel(handlers["basis"])
                model.load(settings["model"]["model_path"])
                handlers["model"] = model
            except (KeyError, ValueError):
                pass
    if "learning" in settings and "basis" in handlers:
        try:
            reg_params = settings["learning"]["regularizer"]
            learning_model = least_squares.WeightedLinearModel(
                handlers["basis"], **reg_params)
            handlers["learning"] = learning_model
        except (KeyError, ValueError):
            pass
    return handlers
