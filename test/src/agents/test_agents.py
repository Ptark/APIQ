import pytest
from python.src.agents.Handcrafted import Handcrafted
from python.src.agents.RandomActions import RandomActions


# fixture method sets up for the tests
@pytest.fixture(scope="class", params=[
    RandomActions,
    Handcrafted,
])
def agents(request):
    """Returns all agents sequentially"""
    return request.param()


def test_calculate_action(agents):
    action = agents.calculate_action(("0", "0"))
    assert isinstance(action, str)
