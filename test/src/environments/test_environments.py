import pytest

from python.src.environments.AnyOne import AnyOne
from python.src.environments.SpecificOne import SpecificOne
from python.src.environments.BiasedCoinFlip import BiasedCoinFlip
from python.src.environments.DoubleCoinFlip import DoubleCoinFlip
from python.src.environments.Slide import Slide


# fixture method sets up for the tests


@pytest.fixture(scope="class", params=[
    BiasedCoinFlip,
    DoubleCoinFlip,
    Slide,
    AnyOne,
    SpecificOne
])
def environments(request):
    """Returns all sensors sequentially"""
    return request.param()


def test_calculate_percept(environments):
    """Test if calculate_percept returns a tuple with 2 strings"""
    percept = environments.calculate_percept("000")
    assert isinstance(percept[0], str)
    assert isinstance(percept[1], str)

