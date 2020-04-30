from python.src.agents.NNAgent import NNAgent


class NNtanh0(NNAgent):
    """Neural Network based agent with no hidden layers"""

    def __init__(self, training_step: int):
        """Initialize NNAgent with given parameters"""
        super().__init__(training_step, activation_name="tanh", size=[8, 4])
