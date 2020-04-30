from python.src.agents.NNAgent import NNAgent


class NNrelu0(NNAgent):
    """Neural Network based agent with no hidden layers"""

    def __init__(self, training_step: int):
        """Initialize NNAgent with given parameters"""
        super().__init__(training_step, activation_name="relu", size=[4, 3])
