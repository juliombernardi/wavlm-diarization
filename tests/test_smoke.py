import importlib


def test_package_imports():
    assert importlib.import_module('wavlmmsdd') is not None
