import numpy as np
import theano
import theano.tensor as T
import math

from Layer import Layer

class BatchNormLayer(Layer):
"""
This layer is not finished - I'm not sure BN makes a lot of sense sinse in a batch we will have motions at various different parts of the phase and
we do not want to normalise these together. This along with the fact that we are aiming for style modelling suggests instance normalisation may be better
"""

	def __init__(self, input_shape, rng=np.random, epsilon=1e-4):
		self.gamma = theano.shared(np.asarray(rng.uniform(low=-1.0/math.sqrt(input_shape[1]), high=1.0/math.sqrt(input_shape[1]), size=(input_shape[1])), dtype=theano.config.floatX), name='gamma', borrow=True)
		self.beta = theano.shared(np.zeros((input_shape[1]), dtype=theano.config.floatX), name='beta', borrow=True)
		self.epsilon = epsilon
		self.params = [self.gamma, self.beta] 

	def __call__(self, input):
		mean = T.mean(input, axis=0)
		var = T.var(input, axis=0)
		normalized = (input - T.mean)/T.sqrt(var+self.epsilon)
		return self.gamma * normalized + self.beta
