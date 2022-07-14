"""
This module provides functions for serializing and loading nested dictionaries
with JSON files.
"""

import json
from typing import Union
import numpy as np


def dump_interaction_map(interaction_map,
                         indent=4,
                         filename=None,
                         write=False):
    """
    Utility function for writing ragged arrays to json file.
        e.g. {("A", "B"): [[1, 2, 3], [4, 5], [6, 7, 8, 9]]}

    Args:
        interaction_map (dict): map of interaction to ragged array
            containing np.ndarry vectors.
        indent (int): number of spaces to indent.
        filename (str): name of file to write.
        write (bool): whether to write to file.
    """
    formatted_map = encode_interaction_map(interaction_map)
    text = json.dumps(formatted_map,
                      indent=indent,
                      cls=CompactJSONEncoder)
    if write:
        with open(filename, 'w') as f:
            f.write(text)
    else:
        return text


def encode_interaction_map(interaction_map):
    """Recursive function for converting arrays to lists and
    tuples into dash-joined keys for JSON serialization."""
    encoded_map = {}
    for key, value in interaction_map.items():
        if isinstance(value, list):  # array to list
            if isinstance(value[0], np.ndarray):
                value = [entry.tolist() for entry in value]
        if isinstance(value, np.ndarray):  # array to list
            value = value.tolist()
        elif isinstance(value, dict):
            value = encode_interaction_map(value)
        if isinstance(key, tuple):  # tuple to joined str
            key = '-'.join([str(item) for item in key])
        encoded_map[key] = value
    return encoded_map


def load_interaction_map(filename):
    """Parse interaction map(s) from JSON."""
    with open(filename, "r") as f:
        formatted_map = json.load(f)
    interaction_map = decode_interaction_map(formatted_map)
    return interaction_map


def decode_interaction_map(formatted_map):
    """Recursive function for converting lists to arrays and
    dash-joined keys into tuples for JSON deserialization."""
    decoded_map = {}
    for key, value in formatted_map.items():
        if isinstance(value, list):  # list to array
            if isinstance(value[0], list):
                value = [np.array(row) for row in value]
            else:
                value = np.array(value)
        elif isinstance(value, dict):
            value = decode_interaction_map(value)
        if '-' in key:  # joined str to tuple
            key = key.split('-')
            try:
                key = [int(i) for i in key]
            except ValueError:
                pass
            key = tuple(key)
        decoded_map[key] = value
    return decoded_map


class CompactJSONEncoder(json.JSONEncoder):
    """
    A JSON Encoder that formats vectors into single lines.

    Discussion on StackOverflow:
        https://stackoverflow.com/questions/16264515/
    Original question by Saar Drimer:
        https://stackoverflow.com/users/458116/saar-drimer
    Original answer by Tim Ludwinski:
        https://stackoverflow.com/users/1413201/tim-ludwinski
    Adaptation by Jannis Mainczyk:
        https://stackoverflow.com/users/5379377/jmm

    Adapted code under CC BY-SA 3.0 license.
    """

    CONTAINER_TYPES = (list, tuple, dict)
    """Container datatypes include primitives or other containers."""

    INDENTATION_CHAR = " "

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.indentation_level = 0

    def encode(self, o):
        """Encode JSON object *o* with respect to single line lists."""
        if isinstance(o, (list, tuple)):
            if self._put_on_single_line(o):
                return "[" + ", ".join(self.encode(el) for el in o) + "]"
            else:
                self.indentation_level += 1
                output = [self.indent_str + self.encode(el) for el in o]
                self.indentation_level -= 1
                return ("[\n" +
                        ",\n".join(output) + "\n" + self.indent_str + "]")
        elif isinstance(o, dict):
            if o:
                if self._put_on_single_line(o):
                    return ("{ " +
                            ", ".join(f"{self.encode(k)}: {self.encode(el)}"
                                      for k, el in o.items()) + " }")
                else:
                    self.indentation_level += 1
                    output = [self.indent_str
                              + f"{json.dumps(k)}: {self.encode(v)}"
                              for k, v in o.items()]
                    self.indentation_level -= 1
                    return ("{\n" +
                            ",\n".join(output) + "\n" + self.indent_str + "}")
            else:
                return "{}"
        elif isinstance(o, float):
            # Use scientific notation for floats, where appropiate
            return format(o, ".17g")
        elif isinstance(o, str):  # escape newlines
            o = o.replace("\n", "\\n")
            return f'"{o}"'
        else:
            return json.dumps(o)

    def _put_on_single_line(self, o):
        return self._primitives_only(o)

    def _primitives_only(self, o: Union[list, tuple, dict]):
        if isinstance(o, (list, tuple)):
            return not any(isinstance(el, self.CONTAINER_TYPES)
                           for el in o)
        elif isinstance(o, dict):
            return not any(isinstance(el, self.CONTAINER_TYPES)
                           for el in o.values())

    @property
    def indent_str(self) -> str:
        return self.INDENTATION_CHAR * (self.indentation_level * self.indent)
