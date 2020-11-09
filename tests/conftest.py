""" Basic configuration for testing """
import pytest
from pathlib import Path


@pytest.fixture(scope="session")
def test_files():
    return Path(__file__).parent.parent.joinpath("test_files")


def pytest_itemcollected(item):
    """Make tests names more readable in the tests output."""
    item._nodeid = (
        item._nodeid.replace(".py", "")
        .replace("tests/", "")
        .replace("test_", "")
        .replace("_", " ")
        .replace("Test", "")
        .replace("Class", " class")
        .lower()
    )
    doc = item.obj.__doc__.strip() if item.obj.__doc__ else ""
    if doc:
        item._nodeid = item._nodeid.split("::")[0] + "::" + doc
