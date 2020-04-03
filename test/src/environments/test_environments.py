import pytest

from python.src.environments.Slide import Slide
from python.src.environments.BiasedCoinFlip import BiasedCoinFlip
from python.src.environments.BiasedCoinFlipReversed import BiasedCoinFlipReversed
from python.src.environments.DoubleCoinFlip import DoubleCoinFlip
from python.src.environments.DoubleCoinFlipReversed import DoubleCoinFlipReversed


# fixture method sets up for the tests
from python.src.environments.SlideReversed import SlideReversed


@pytest.fixture(scope="class", params=[
    BiasedCoinFlip,
    BiasedCoinFlipReversed,
    DoubleCoinFlip,
    DoubleCoinFlipReversed,
    Slide,
    SlideReversed
])
def environments(request):
    """Returns all sensors sequentially"""
    return request.param()


def test_calculate_percept(environments):
    percept = environments.calculate_percept((0, 0, 0, 0))
    assert len(percept) == 2

