import pytest

from python.src.agents.RandomAgent import RandomAgent


@pytest.fixture
def random_agent():
    return RandomAgent()


def test_calculate_action(random_agent):
    action = random_agent.calculate_action(((0, 0, 0, 0), (0, 0, 0, 0)))
    assert len(action) == 4
