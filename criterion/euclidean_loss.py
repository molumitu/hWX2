""" Euclidean Loss Layer """

import numpy as np

class EuclideanLossLayer():
    def __init__(self):
        self.acc = 0.
        self.loss = 0.
        self.logit = None
        self.logit = None

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
        # batch_size = np.array(gt).reshape(-1)
        # gt_onehot = np.eye(10)[targets]

        self.logit = logit
        self.gt = gt
        batch_size = np.shape(logit)[0]
        num_acc = (batch_size*10 - np.sum(gt == logit))/2
        self.acc = num_acc/batch_size
        #self.loss = 0.5 * np.sum(np.square(gt - logit))
        self.loss = ((logit - gt) ** 2).mean(axis=0).sum() / 2
        ############################################################################

        return self.loss

    def backward(self):

        ############################################################################
        # TODO: Put your code here
        # Calculate and return the gradient (have the same shape as logit)
        return self.logit - self.gt

        ############################################################################
