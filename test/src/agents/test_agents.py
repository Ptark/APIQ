import pytest
from python.src.agents.HandcraftedAgent import HandcraftedAgent
from python.src.agents.RandomAgent import RandomAgent


# fixture method sets up for the tests
@pytest.fixture(scope="class", params=[
    RandomAgent,
    HandcraftedAgent,
])
def agents(request):
    """Returns all agents sequentially"""
    return RandomAgent()


def test_calculate_action(agents):
    action = agents.calculate_action(("0", "0"))
    assert isinstance(action, str)
