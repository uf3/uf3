import os
import re
import yaml
import uf3
from ase import symbols as ase_symbols

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
                                  "default_config.yaml")
    with open(default_config, "r") as f:
        settings = yaml.load(f, Loader=yaml.Loader)
    with open(settings_filename, "r") as f:
        user_settings = yaml.load(f, Loader=yaml.Loader)
    for k in user_settings:
        v = user_settings[k]
        if k in settings:
            type_target = type(settings[k])
            type_user = type(v)
            if type_target == bool:  # boolean
                settings[k] = bool(v)
            if type_target in [float, int]:  # number
                if type_user in [float, int]:
                    settings[k] = type_target(v)
            elif type_target in [list, tuple]:  # iterable
                if type_user in [list, tuple]:
                    settings[k] = list(v)
            elif type_target == type_user:  # other
                settings[k] = v
        else:
            settings[k] = v
    return settings