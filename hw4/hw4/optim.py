from typing import Iterable

import numpy as np
from edugrad.optim import Optimizer, SGD
from edugrad.tensor import Tensor

class Adagrad(Optimizer):

    def __init__(self, params: Iterable[Tensor], lr: float = 1e-2):
        super(Adagrad, self).__init__(params)
        self.lr = lr
        self._eps = 1e-7
        # initialize squared gradient history for each param
        for param in self.params:
            param._grad_hist = np.zeros(param.value.shape)

    def step(self):
        for param in self.params:
            # TODO: implement here
            # increment squared gradient history; param.grad contains the gradient
            # get adjusted learning rate
            # update parameters
            param._grad_hist += np.square(param.grad)
            denom = np.sqrt(param._grad_hist + self.eps)
            param.value -= (self.lr/denom) * param.grad
        self._cur_step += 1