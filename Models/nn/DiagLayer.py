import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from Layer import Layer

class DiagLayer(Layer):

    def __init__(self, weights_shape, rng=np.random, gamma=0.01):

        assert weights_shape[-2] == 1 # Diagonal weight matrix is the same as taking a vector of size input and doing an element wise multiplciation

        W_bound = np.sqrt(6. / np.prod(weights_shape[-2:]))
        W = np.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=weights_shape),
            dtype=theano.config.floatX)

        self.W = theano.shared(name='W', value=W, borrow=True)
        self.params = [self.W]
        self.gamma = gamma
        
    def cost(self, input):
        return self.gamma * T.mean(abs(self.W))
        
    def __call__(self, input):
        return self.W * input  # elementwise multiplication 

        

        