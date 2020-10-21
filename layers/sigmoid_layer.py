""" Sigmoid Layer """

import numpy as np

class SigmoidLayer():
    def __init__(self):
        """
        Applies the element-wise function: f(x) = 1/(1+exp(-x))
        """
        self.trainable = False
        self.tensor1 = None



    def forward(self, Input):

        ############################################################################
        # TODO: Put your code here
        # Apply Sigmoid activation function to Input, and return results.
        self.tensor1 = 1. / (1. + np.exp(-Input))
        return self.tensor1

        ############################################################################

    def backward(self, delta):

        ############################################################################
        # TODO: Put your code here
        # Calculate the gradient using the later layer's gradient: delta
        return self.tensor1 * (1. - self.tensor1) * delta

        ############################################################################
