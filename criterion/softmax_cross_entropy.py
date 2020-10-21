""" Softmax Cross-Entropy Loss Layer """

import numpy as np

# a small number to prevent dividing by zero, maybe useful for you
EPS = 1e-11

class SoftmaxCrossEntropyLossLayer():
    def __init__(self):
        self.acc = 0.
        self.loss = np.zeros(1, dtype='f')
        self.batch_size = None
        self.gt_onehot = None
        self.logit_softmax = None
       

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
        self.batch_size = np.shape(logit)[0]
        
        temp1 = np.exp(logit)
        logit_softmax = temp1 / np.sum(temp1, axis=1)[:, None]

        logit_softmax_index = np.argmax(logit_softmax, axis=1)
        targets = logit_softmax_index.reshape(-1)
        logit_softmax_onehot = np.eye(10)[targets]
        
        
        #loss = -np.sum(label * np.log(f_result))/batch_size 

        self.logit_softmax = logit_softmax
        self.gt_onehot = gt


        num_error = (self.batch_size*10 - np.sum(self.gt_onehot == logit_softmax_onehot))/2
        self.acc = 1 - num_error/self.batch_size
        self.loss = -np.sum(self.gt_onehot * np.log(self.logit_softmax))/self.batch_size
        ############################################################################

        return self.loss


    def backward(self):

        ############################################################################
        # TODO: Put your code here
        # Calculate and return the gradient (have the same shape as logit)
        return (self.logit_softmax - self.gt_onehot)/self.batch_size

        ############################################################################
