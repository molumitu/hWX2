""" Euclidean Loss Layer """

import numpy as np

class EuclideanLossLayer():
	def __init__(self):
		self.accu = 0.
		self.loss = 0.

	def forward(self, logit, gt):
		"""
	      Inputs: (minibatch)
	      - logit: forward results from the last FCLayer, shape(batch_size, 10)
	      - gt: the ground truth label, shape(batch_size, 1)
	    """

		############################################################################
	    # TODO: Put your code here
		# Calculate the average accuracy and loss over the minibatch, and
		# store in self.accu and self.loss respectively.
		# Only return the self.loss, self.accu will be used in solver.py.
    batch_size = np.array(gt).reshape(-1)
    gt_onehot = np.eye(10)[targets]


    num_total = np.shape(logit)[0]
    num_accu = (batch_size*10 - np.sum(gt_onehot == logit))/2
    self.accu = num_acc/batch_size
    self.loss = 0.5 * np.sum(np.square(gt_onehot - logit))
	    ############################################################################

		return self.loss

	def backward(self):

		############################################################################
	    # TODO: Put your code here
		# Calculate and return the gradient (have the same shape as logit)


	    ############################################################################
