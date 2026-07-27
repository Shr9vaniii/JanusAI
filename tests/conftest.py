import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: tests that need corpus/network/GPU")
