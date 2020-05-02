from python.src.agents.NNAgent import NNAgent


class NNrelu4443(NNAgent):
    """Neural Network based agent with hidden layers"""

    def __init__(self, training_step: int):
        """Initialize NNAgent with given parameters"""
        super().__init__(training_step, activation_name="relu", size=[4, 4, 4, 3])
