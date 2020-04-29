from python.src.agents.NNAgent import NNAgent


class NN0x0relu0(NNAgent):
    """Neural Network based agent with 0 training steps and a 0x0 hidden layer"""

    def __init__(self):
        """Initialize NNAgent with given parameters"""
        super().__init__(activation_name="relu", size=[8, 4], training_steps=0)
