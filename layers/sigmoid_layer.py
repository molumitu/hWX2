""" Sigmoid Layer """

import numpy as np

class SigmoidLayer():
	def __init__(self):
		"""
		Applies the element-wise function: f(x) = 1/(1+exp(-x))
		"""
		self.trainable = False
    self.output = None

	def forward(self, Input):

		############################################################################
	    # TODO: Put your code here
		# Apply Sigmoid activation function to Input, and return results.
    self.output = 1/(1 + np.exp(Input))
    return self.output

	    ############################################################################

	def backward(self, delta):

		############################################################################
	    # TODO: Put your code here
		# Calculate the gradient using the later layer's gradient: delta
    return delta @ self.output @ (1 - self.output)

	    ############################################################################
