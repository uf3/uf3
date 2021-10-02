import os
import yaml
import uf3
from uf3.util import user_config


def test_type_check():
    assert type(user_config.type_check(1.0, False)) == bool
    assert type(user_config.type_check(1.0, 2)) == int
    assert type(user_config.type_check(3, 4.0)) == float
    assert type(user_config.type_check("5", 6)) == int
    assert type(user_config.type_check("7", 8.0)) == float
    assert type(user_config.type_check((1, 2, 3), (4, 5, 6))) == list
    assert type(user_config.type_check(dict(a=2, b=4), dict(a=3, b=5))) == dict
    assert type(user_config.type_check(9, None)) == int


def test_consistency_check():
    ref = dict(a=dict(a=1, b=2, c=3, d=4, e=5, f=6), b=2)
    user = dict(a=dict(e=50, f=60))
    consistent = user_config.consistency_check(user, ref)
    print(consistent)
    assert len(consistent) == 2
    assert len(consistent["a"]) == 6
    assert consistent["a"]["e"] != 5


def test_read_config():
    default_config = os.path.join(os.path.dirname(uf3.__file__),
                                  "default_options.yaml")
    settings = user_config.read_config(default_config)
    assert all([key in settings for key
                in ["data", "basis", "features", "model", "learning"]])


def test_generate_handlers():
    filename = os.path.join(os.path.dirname(uf3.__file__),
                            "default_options.yaml")
    settings = user_config.read_config(filename)
    settings["elements"] = ["W"]
    handlers = user_config.generate_handlers(settings)
    for key in ["data", "basis", "features", "learning"]:
        assert key in handlers
