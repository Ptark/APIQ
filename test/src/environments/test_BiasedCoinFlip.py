import pytest

from python.src.environments.BiasedCoinFlip import BiasedCoinFlip


@pytest.fixture
def biased_coin_flip():
    return BiasedCoinFlip()


def test_calculate_percept(biased_coin_flip):
    percept = biased_coin_flip.calculate_percept((0, 0, 0, 0))
    assert len(percept) == 2

