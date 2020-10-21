""" Softmax Cross-Entropy Loss Layer """

import numpy as np

# a small number to prevent dividing by zero, maybe useful for you
EPS = 1e-11

class SoftmaxCrossEntropyLossLayer():
    def __init__(self):
        self.acc = 0.
        self.loss = np.zeros(1, dtype='f')
       

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
        targets = np.array(gt).reshape(-1)
        gt_onehot = np.eye(10)[targets]

        batch_size = np.shape(logit)[0]
        num_accu = (batch_size*10 - np.sum(gt_onehot == logit))/2
        self.accu = num_acc/batch_size
        self.loss = -np.sum(logit * np.log(gt_onehot))/num_total
        ############################################################################

        return self.loss


    def backward(self):

        ############################################################################
        # TODO: Put your code here
        # Calculate and return the gradient (have the same shape as logit)
        return (gt - logit) 

        ############################################################################
