""" Softmax Cross-Entropy Loss Layer """

import numpy as np

# a small number to prevent dividing by zero, maybe useful for you
EPS = 1e-11

class SoftmaxCrossEntropyLossLayer():
    def __init__(self):
        self.acc = 0.
        self.loss = np.zeros(1, dtype='f')
        self.batch_size = None
       

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
        # targets = np.array(gt).reshape(-1)
        # gt_onehot = np.eye(10)[targets]
        logit_index = np.argmax(logit, axis=1)
        targets = logit_index.reshape(-1)
        logit_onehot = np.eye(10)[targets]

        self.logit_onehot = logit_onehot
        self.gt_onehot = gt

        self.batch_size = np.shape(logit)[0]
        num_error = (self.batch_size*10 - np.sum(gt_onehot == logit))/2
        self.acc = 1 - num_error/self.batch_size
        self.loss = -np.sum(logit * np.log(gt_onehot))/self.batch_size
        ############################################################################

        return self.loss


    def backward(self):

        ############################################################################
        # TODO: Put your code here
        # Calculate and return the gradient (have the same shape as logit)
        return (self.logit_onehot - gt_onehot) / self.batch_size

        ############################################################################
