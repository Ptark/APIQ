from python.src.agents.NNAgent import NNAgent


class NNsigmoid0(NNAgent):
    """Neural Network based agent with no hidden layers"""

    def __init__(self, training_step: int):
        """Initialize NNAgent with given parameters"""
        super().__init__(training_step, activation_name="sigmoid", size=[4, 3])
