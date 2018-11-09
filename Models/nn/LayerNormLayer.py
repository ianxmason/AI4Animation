import numpy as np
import theano
import theano.tensor as T
import math

from Layer import Layer

class LayerNormLayer(Layer):
	"""
	Instance normalisation makes no sense for non-spatial data as we have, so instead we turn to the layer normalisation of https://arxiv.org/pdf/1607.06450.pdf
	Note: a summary of BN vs IN vs CIN is here https://arxiv.org/pdf/1703.06868.pdf
	"""
	def __init__(self, input_shape, rng=np.random, epsilon=1e-5):
		self.gamma = theano.shared(np.asarray(rng.uniform(low=-1.0/math.sqrt(input_shape[1]), high=1.0/math.sqrt(input_shape[1]), size=input_shape), dtype=theano.config.floatX), name='gamma', borrow=True)
		self.beta = theano.shared(np.zeros(input_shape, dtype=theano.config.floatX), name='beta', borrow=True)
		self.epsilon = epsilon
		self.params = [self.gamma, self.beta] 

	def __call__(self, input):
		mean = T.mean(input, axis=1, keepdims=True) # normalise over the features rather than over the batch
		var = T.var(input, axis=1, keepdims=True)
		normalized = (input - mean)/T.sqrt(var+self.epsilon)
		return self.gamma * normalized + self.beta
