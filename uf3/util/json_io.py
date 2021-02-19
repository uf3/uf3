import json
from typing import Union
import numpy as np


def dump_interaction_map(interaction_map,
                         indent=4,
                         filename="interaction.json",
                         write=True):
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
    map_copy = {}
    for key, value in interaction_map.items():
        if isinstance(key, tuple):
            key = '-'.join([str(item) for item in key])
            # tuple keys must be converted for json
        if isinstance(value, np.ndarray):
            map_copy[key] = value.tolist()
        else:
            map_copy[key] = value
        text = json.dumps(map_copy,
                          indent=indent,
                          cls=CompactJSONEncoder)
    if write:
        with open(filename, 'w') as f:
            f.write(text)
    else:
        return text


def load_interaction_map(filename):
    """
    Utility function for reading ragged arrays from json file.
        e.g. {("A", "B"): [[1, 2, 3], [4, 5], [6, 7, 8, 9]]}
    """
    with open(filename, "r") as f:
        formatted_map = json.load(f)
    interaction_map = {}
    for key, value in formatted_map.items():
        if '-' in key:
            key = key.split('-')
            try:
                key = [int(i) for i in key]
            except ValueError:
                pass
            key = tuple(key)
        if isinstance(value, list):
            value = np.array(value)
        interaction_map[key] = value
    return interaction_map


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
            return format(o, "g")
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
