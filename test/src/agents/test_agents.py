import pytest
from python.src.agents.Handcrafted import HandcraftedAgent
from python.src.agents.RandomActions import RandomAgent


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
