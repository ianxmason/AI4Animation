""" 
Training the main pfnn with residual adapters with a full matrix of weights for each style.
"""

import sys
import numpy as np
import theano
import theano.tensor as T
import time
theano.config.allow_gc = True
sys.path.append('./nn')
from Layer import Layer
from HiddenLayer import HiddenLayer
from BiasLayer import BiasLayer
from DropoutLayer import DropoutLayer
from ActivationLayer import ActivationLayer
from AdamTrainerStyle import AdamTrainer

mname='Full'
rng = np.random.RandomState(23456)

""" Load Data """

database = np.load('../Style_Data/style_database3.npz')
X_default = database['Xin']
Y_default = database['Yin'].astype(theano.config.floatX)
P_default = database['Pin'].astype(theano.config.floatX)
X_mirror = database['Xin_mirror']
Y_mirror = database['Yin_mirror'].astype(theano.config.floatX)
P_mirror = database['Pin_mirror'].astype(theano.config.floatX)

j = 31
w = ((60*2)//10)

L_default = np.copy(X_default[:,w*6:w*14]).astype(theano.config.floatX)
L_mirror = np.copy(X_mirror[:,w*6:w*14]).astype(theano.config.floatX)
X_default = np.concatenate((X_default[:,w*0:w*4], X_default[:,w*14:]), axis=1).astype(theano.config.floatX)
X_mirror = np.concatenate((X_mirror[:,w*0:w*4], X_mirror[:,w*14:]), axis=1).astype(theano.config.floatX)

print(X_default.shape, Y_default.shape, L_default.shape, P_default.shape)
print(X_mirror.shape, Y_mirror.shape, L_mirror.shape, P_mirror.shape)

X = np.concatenate((X_default, X_mirror), axis=0)
Y = np.concatenate((Y_default, Y_mirror), axis=0)
P = np.concatenate((P_default, P_mirror), axis=0)
L = np.concatenate((L_default, L_mirror), axis=0)

print(X.shape, Y.shape, L.shape, P.shape)

""" Calculate Mean and Std """

Xmean, Xstd = X.mean(axis=0), X.std(axis=0)
Ymean, Ystd = Y.mean(axis=0), Y.std(axis=0)


Xstd[w*0:w*1] = Xstd[w*0:w*1].mean() # Trajectory X Positions
Xstd[w*1:w*2] = Xstd[w*1:w*2].mean() # Trajectory Z Positions
Xstd[w*2:w*3] = Xstd[w*2:w*3].mean() # Trajectory X Directions
Xstd[w*3:w*4] = Xstd[w*3:w*4].mean() # Trajectory Z Directions

""" Mask Out Unused Joints in Input """

joint_weights = np.array([
    1,
    1e-10, 1, 1, 1, 1,
    1e-10, 1, 1, 1, 1,
    1e-10, 1, 1,
    1e-10, 1, 1,
    1e-10, 1, 1, 1, 1e-10, 1e-10, 1e-10,
    1e-10, 1, 1, 1, 1e-10, 1e-10, 1e-10]).repeat(3)

Xstd[w*4+j*3*0:w*4+j*3*1] = Xstd[w*4+j*3*0:w*4+j*3*1].mean() / (joint_weights * 0.1) # Pos
Xstd[w*4+j*3*1:w*4+j*3*2] = Xstd[w*4+j*3*1:w*4+j*3*2].mean() / (joint_weights * 0.1) # Vel
assert w*4+j*3*2 == X.shape[1]

Ystd[0:2] = Ystd[0:2].mean() # Translational Velocity
Ystd[2:3] = Ystd[2:3].mean() # Rotational Velocity
Ystd[3:4] = Ystd[3:4].mean() # Change in Phase

Ystd[4+w*0:4+w*1] = Ystd[4+w*0:4+w*1].mean() # Trajectory Future Positions
Ystd[4+w*1:4+w*2] = Ystd[4+w*1:4+w*2].mean() # Trajectory Future Directions

Ystd[4+w*2+j*3*0:4+w*2+j*3*1] = Ystd[4+w*2+j*3*0:4+w*2+j*3*1].mean() # Pos
Ystd[4+w*2+j*3*1:4+w*2+j*3*2] = Ystd[4+w*2+j*3*1:4+w*2+j*3*2].mean() # Vel
Ystd[4+w*2+j*3*2:4+w*2+j*3*3] = Ystd[4+w*2+j*3*2:4+w*2+j*3*3].mean() # Fwd Rot
Ystd[4+w*2+j*3*3:4+w*2+j*3*4] = Ystd[4+w*2+j*3*3:4+w*2+j*3*4].mean() # Up Rot

""" Save Mean / Std / Min / Max """

Xmean.astype(np.float32).tofile('./Parameters/' + mname + '/Xmean.bin')
Ymean.astype(np.float32).tofile('./Parameters/' + mname + '/Ymean.bin')
Xstd.astype(np.float32).tofile('./Parameters/' + mname + '/Xstd.bin')
Ystd.astype(np.float32).tofile('./Parameters/' + mname + '/Ystd.bin')

""" Normalize Data """

X_default = (X_default - Xmean) / Xstd
Y_default = (Y_default - Ymean) / Ystd
X_mirror = (X_mirror - Xmean) / Xstd
Y_mirror = (Y_mirror - Ymean) / Ystd


""" Phase Function Neural Network """

class PhaseFunctionedNetwork(Layer):
    
    def __init__(self, rng=rng, input_shape=1, output_shape=1, dropout=0.7):
        
        self.nslices = 4        
        self.dropout0 = DropoutLayer(dropout, rng=rng)
        self.dropout1 = DropoutLayer(dropout, rng=rng)
        self.dropout2 = DropoutLayer(dropout, rng=rng)
        self.activation = ActivationLayer('ELU')
        
        self.W0 = HiddenLayer((self.nslices, 512, input_shape-1), rng=rng, gamma=0.01)
        self.W1 = HiddenLayer((self.nslices, 512, 512), rng=rng, gamma=0.01)
        self.W2 = HiddenLayer((self.nslices, output_shape, 512), rng=rng, gamma=0.01)
    
        self.b0 = BiasLayer((self.nslices, 512))
        self.b1 = BiasLayer((self.nslices, 512))
        self.b2 = BiasLayer((self.nslices, output_shape))


        self.ang_W = HiddenLayer((self.nslices, 512, 512), rng=rng, gamma=0.01)
        self.ang_b = BiasLayer((self.nslices, 512))
        self.chi_W = HiddenLayer((self.nslices, 512, 512), rng=rng, gamma=0.01)
        self.chi_b = BiasLayer((self.nslices, 512))
        self.dep_W = HiddenLayer((self.nslices, 512, 512), rng=rng, gamma=0.01)
        self.dep_b = BiasLayer((self.nslices, 512))
        self.neu_W = HiddenLayer((self.nslices, 512, 512), rng=rng, gamma=0.01)
        self.neu_b = BiasLayer((self.nslices, 512))
        self.old_W = HiddenLayer((self.nslices, 512, 512), rng=rng, gamma=0.01)
        self.old_b = BiasLayer((self.nslices, 512))
        self.pro_W = HiddenLayer((self.nslices, 512, 512), rng=rng, gamma=0.01)
        self.pro_b = BiasLayer((self.nslices, 512))
        self.sex_W = HiddenLayer((self.nslices, 512, 512), rng=rng, gamma=0.01)
        self.sex_b = BiasLayer((self.nslices, 512))
        self.str_W = HiddenLayer((self.nslices, 512, 512), rng=rng, gamma=0.01)
        self.str_b = BiasLayer((self.nslices, 512))

        self.layers = [
            self.W0, self.W1, self.W2,
            self.b0, self.b1, self.b2,
            self.ang_W, self.ang_b,
            self.chi_W, self.chi_b,
            self.dep_W, self.dep_b,
            self.neu_W, self.neu_b,
            self.old_W, self.old_b,
            self.pro_W, self.pro_b,
            self.sex_W, self.sex_b,
            self.str_W, self.str_b]

        self.params = sum([layer.params for layer in self.layers], [])

        ang_label = np.zeros(L.shape[1])
        ang_label[w*0:w*1] = 1
        chi_label = np.zeros(L.shape[1])
        chi_label[w*1:w*2] = 1
        dep_label = np.zeros(L.shape[1])
        dep_label[w*2:w*3] = 1
        neu_label = np.zeros(L.shape[1])
        neu_label[w*3:w*4] = 1
        old_label = np.zeros(L.shape[1])
        old_label[w*4:w*5] = 1
        pro_label = np.zeros(L.shape[1])
        pro_label[w*5:w*6] = 1
        sex_label = np.zeros(L.shape[1])
        sex_label[w*6:w*7] = 1
        str_label = np.zeros(L.shape[1])
        str_label[w*7:w*8] = 1

        self.ang_label = theano.shared(ang_label, borrow=True)
        self.chi_label = theano.shared(chi_label, borrow=True)
        self.dep_label = theano.shared(dep_label, borrow=True)
        self.neu_label = theano.shared(neu_label, borrow=True)
        self.old_label = theano.shared(old_label, borrow=True)
        self.pro_label = theano.shared(pro_label, borrow=True)
        self.sex_label = theano.shared(sex_label, borrow=True)
        self.str_label = theano.shared(str_label, borrow=True)

        zeros = np.zeros((1, output_shape))
        self.zeros = T.addbroadcast(theano.shared(zeros, borrow=True), 0)
        
    def __call__(self, input, label):
        
        pscale = self.nslices * input[:,-1]
        pamount = pscale % 1.0
        
        pindex_1 = T.cast(pscale, 'int32') % self.nslices
        pindex_0 = (pindex_1-1) % self.nslices
        pindex_2 = (pindex_1+1) % self.nslices
        pindex_3 = (pindex_1+2) % self.nslices
        
        Wamount = pamount.dimshuffle(0, 'x', 'x')
        bamount = pamount.dimshuffle(0, 'x')
        
        def cubic(y0, y1, y2, y3, mu):
            return (
                (-0.5*y0+1.5*y1-1.5*y2+0.5*y3)*mu*mu*mu + 
                (y0-2.5*y1+2.0*y2-0.5*y3)*mu*mu + 
                (-0.5*y0+0.5*y2)*mu +
                (y1))
        
        W0 = cubic(self.W0.W[pindex_0], self.W0.W[pindex_1], self.W0.W[pindex_2], self.W0.W[pindex_3], Wamount)
        W1 = cubic(self.W1.W[pindex_0], self.W1.W[pindex_1], self.W1.W[pindex_2], self.W1.W[pindex_3], Wamount)
        W2 = cubic(self.W2.W[pindex_0], self.W2.W[pindex_1], self.W2.W[pindex_2], self.W2.W[pindex_3], Wamount)
        
        b0 = cubic(self.b0.b[pindex_0], self.b0.b[pindex_1], self.b0.b[pindex_2], self.b0.b[pindex_3], bamount)
        b1 = cubic(self.b1.b[pindex_0], self.b1.b[pindex_1], self.b1.b[pindex_2], self.b1.b[pindex_3], bamount)
        b2 = cubic(self.b2.b[pindex_0], self.b2.b[pindex_1], self.b2.b[pindex_2], self.b2.b[pindex_3], bamount)
        
        ang_W = cubic(self.ang_W.W[pindex_0], self.ang_W.W[pindex_1], self.ang_W.W[pindex_2], self.ang_W.W[pindex_3], Wamount)
        ang_b = cubic(self.ang_b.b[pindex_0], self.ang_b.b[pindex_1], self.ang_b.b[pindex_2], self.ang_b.b[pindex_3], bamount)
        chi_W = cubic(self.chi_W.W[pindex_0], self.chi_W.W[pindex_1], self.chi_W.W[pindex_2], self.chi_W.W[pindex_3], Wamount)
        chi_b = cubic(self.chi_b.b[pindex_0], self.chi_b.b[pindex_1], self.chi_b.b[pindex_2], self.chi_b.b[pindex_3], bamount)
        dep_W = cubic(self.dep_W.W[pindex_0], self.dep_W.W[pindex_1], self.dep_W.W[pindex_2], self.dep_W.W[pindex_3], Wamount)
        dep_b = cubic(self.dep_b.b[pindex_0], self.dep_b.b[pindex_1], self.dep_b.b[pindex_2], self.dep_b.b[pindex_3], bamount)
        neu_W = cubic(self.neu_W.W[pindex_0], self.neu_W.W[pindex_1], self.neu_W.W[pindex_2], self.neu_W.W[pindex_3], Wamount)
        neu_b = cubic(self.neu_b.b[pindex_0], self.neu_b.b[pindex_1], self.neu_b.b[pindex_2], self.neu_b.b[pindex_3], bamount)
        old_W = cubic(self.old_W.W[pindex_0], self.old_W.W[pindex_1], self.old_W.W[pindex_2], self.old_W.W[pindex_3], Wamount)
        old_b = cubic(self.old_b.b[pindex_0], self.old_b.b[pindex_1], self.old_b.b[pindex_2], self.old_b.b[pindex_3], bamount)
        pro_W = cubic(self.pro_W.W[pindex_0], self.pro_W.W[pindex_1], self.pro_W.W[pindex_2], self.pro_W.W[pindex_3], Wamount)
        pro_b = cubic(self.pro_b.b[pindex_0], self.pro_b.b[pindex_1], self.pro_b.b[pindex_2], self.pro_b.b[pindex_3], bamount)
        sex_W = cubic(self.sex_W.W[pindex_0], self.sex_W.W[pindex_1], self.sex_W.W[pindex_2], self.sex_W.W[pindex_3], Wamount)
        sex_b = cubic(self.sex_b.b[pindex_0], self.sex_b.b[pindex_1], self.sex_b.b[pindex_2], self.sex_b.b[pindex_3], bamount)
        str_W = cubic(self.str_W.W[pindex_0], self.str_W.W[pindex_1], self.str_W.W[pindex_2], self.str_W.W[pindex_3], Wamount)
        str_b = cubic(self.str_b.b[pindex_0], self.str_b.b[pindex_1], self.str_b.b[pindex_2], self.str_b.b[pindex_3], bamount)

        H0 = input[:,:-1]
        H1 = self.activation(T.batched_dot(W0, self.dropout0(H0)) + b0)

        ang_H3 = T.switch(T.all(T.eq(label[0],self.ang_label)), T.batched_dot(W2, self.dropout2(self.activation(T.batched_dot(W1, self.dropout1(H1)) + b1 + T.batched_dot(ang_W, self.dropout1(H1)) + ang_b))) + b2, self.zeros)
        chi_H3 = T.switch(T.all(T.eq(label[0],self.chi_label)), T.batched_dot(W2, self.dropout2(self.activation(T.batched_dot(W1, self.dropout1(H1)) + b1 + T.batched_dot(chi_W, self.dropout1(H1)) + chi_b))) + b2, self.zeros)
        dep_H3 = T.switch(T.all(T.eq(label[0],self.dep_label)), T.batched_dot(W2, self.dropout2(self.activation(T.batched_dot(W1, self.dropout1(H1)) + b1 + T.batched_dot(dep_W, self.dropout1(H1)) + dep_b))) + b2, self.zeros)
        neu_H3 = T.switch(T.all(T.eq(label[0],self.neu_label)), T.batched_dot(W2, self.dropout2(self.activation(T.batched_dot(W1, self.dropout1(H1)) + b1 + T.batched_dot(neu_W, self.dropout1(H1)) + neu_b))) + b2, self.zeros)
        old_H3 = T.switch(T.all(T.eq(label[0],self.old_label)), T.batched_dot(W2, self.dropout2(self.activation(T.batched_dot(W1, self.dropout1(H1)) + b1 + T.batched_dot(old_W, self.dropout1(H1)) + old_b))) + b2, self.zeros)
        pro_H3 = T.switch(T.all(T.eq(label[0],self.pro_label)), T.batched_dot(W2, self.dropout2(self.activation(T.batched_dot(W1, self.dropout1(H1)) + b1 + T.batched_dot(pro_W, self.dropout1(H1)) + pro_b))) + b2, self.zeros)
        sex_H3 = T.switch(T.all(T.eq(label[0],self.sex_label)), T.batched_dot(W2, self.dropout2(self.activation(T.batched_dot(W1, self.dropout1(H1)) + b1 + T.batched_dot(sex_W, self.dropout1(H1)) + sex_b))) + b2, self.zeros)
        str_H3 = T.switch(T.all(T.eq(label[0],self.str_label)), T.batched_dot(W2, self.dropout2(self.activation(T.batched_dot(W1, self.dropout1(H1)) + b1 + T.batched_dot(str_W, self.dropout1(H1)) + str_b))) + b2, self.zeros)

       
        return ang_H3 + chi_H3 + dep_H3 + neu_H3 + old_H3 + pro_H3 + sex_H3 + str_H3
        
    def cost(self, input):
        input = input[:,:-1]
        costs = 0
        for layer in self.layers:
            costs += layer.cost(input)
            input = layer(input)
        return costs / len(self.layers)
    
    def save(self, database, prefix=''):
        for li, layer in enumerate(self.layers):
            layer.save(database, '%sL%03i_' % (prefix, li))
        
    def load(self, database, prefix=''):
        for li, layer in enumerate(self.layers):
            layer.load(database, '%sL%03i_' % (prefix, li))

            
""" Function to Save Network Weights """

def save_network(network):

    """ Load Control Points """

    W0n = network.W0.W.get_value()
    W1n = network.W1.W.get_value()
    W2n = network.W2.W.get_value()

    b0n = network.b0.b.get_value()
    b1n = network.b1.b.get_value()
    b2n = network.b2.b.get_value()

    ang_Wn = network.ang_W.W.get_value()
    ang_bn = network.ang_b.b.get_value()
    chi_Wn = network.chi_W.W.get_value()
    chi_bn = network.chi_b.b.get_value()
    dep_Wn = network.dep_W.W.get_value()
    dep_bn = network.dep_b.b.get_value()
    neu_Wn = network.neu_W.W.get_value()
    neu_bn = network.neu_b.b.get_value()
    old_Wn = network.old_W.W.get_value()
    old_bn = network.old_b.b.get_value()
    pro_Wn = network.pro_W.W.get_value()
    pro_bn = network.pro_b.b.get_value()
    sex_Wn = network.sex_W.W.get_value()
    sex_bn = network.sex_b.b.get_value()
    str_Wn = network.str_W.W.get_value()
    str_bn = network.str_b.b.get_value()
    
    """ Precompute Phase Function """
    
    for i in range(50):
        
        pscale = network.nslices*(float(i)/50)
        pamount = pscale % 1.0
        
        pindex_1 = int(pscale) % network.nslices
        pindex_0 = (pindex_1-1) % network.nslices
        pindex_2 = (pindex_1+1) % network.nslices
        pindex_3 = (pindex_1+2) % network.nslices
        
        def cubic(y0, y1, y2, y3, mu):
            return (
                (-0.5*y0+1.5*y1-1.5*y2+0.5*y3)*mu*mu*mu + 
                (y0-2.5*y1+2.0*y2-0.5*y3)*mu*mu + 
                (-0.5*y0+0.5*y2)*mu +
                (y1))
        
        W0 = cubic(W0n[pindex_0], W0n[pindex_1], W0n[pindex_2], W0n[pindex_3], pamount)
        W1 = cubic(W1n[pindex_0], W1n[pindex_1], W1n[pindex_2], W1n[pindex_3], pamount)
        W2 = cubic(W2n[pindex_0], W2n[pindex_1], W2n[pindex_2], W2n[pindex_3], pamount)
        
        b0 = cubic(b0n[pindex_0], b0n[pindex_1], b0n[pindex_2], b0n[pindex_3], pamount)
        b1 = cubic(b1n[pindex_0], b1n[pindex_1], b1n[pindex_2], b1n[pindex_3], pamount)
        b2 = cubic(b2n[pindex_0], b2n[pindex_1], b2n[pindex_2], b2n[pindex_3], pamount)

        ang_W = cubic(ang_Wn[pindex_0], ang_Wn[pindex_1], ang_Wn[pindex_2], ang_Wn[pindex_3], pamount)
        ang_b = cubic(ang_bn[pindex_0], ang_bn[pindex_1], ang_bn[pindex_2], ang_bn[pindex_3], pamount)
        chi_W = cubic(chi_Wn[pindex_0], chi_Wn[pindex_1], chi_Wn[pindex_2], chi_Wn[pindex_3], pamount)
        chi_b = cubic(chi_bn[pindex_0], chi_bn[pindex_1], chi_bn[pindex_2], chi_bn[pindex_3], pamount)
        dep_W = cubic(dep_Wn[pindex_0], dep_Wn[pindex_1], dep_Wn[pindex_2], dep_Wn[pindex_3], pamount)
        dep_b = cubic(dep_bn[pindex_0], dep_bn[pindex_1], dep_bn[pindex_2], dep_bn[pindex_3], pamount)
        neu_W = cubic(neu_Wn[pindex_0], neu_Wn[pindex_1], neu_Wn[pindex_2], neu_Wn[pindex_3], pamount)
        neu_b = cubic(neu_bn[pindex_0], neu_bn[pindex_1], neu_bn[pindex_2], neu_bn[pindex_3], pamount)
        old_W = cubic(old_Wn[pindex_0], old_Wn[pindex_1], old_Wn[pindex_2], old_Wn[pindex_3], pamount)
        old_b = cubic(old_bn[pindex_0], old_bn[pindex_1], old_bn[pindex_2], old_bn[pindex_3], pamount)
        pro_W = cubic(pro_Wn[pindex_0], pro_Wn[pindex_1], pro_Wn[pindex_2], pro_Wn[pindex_3], pamount)
        pro_b = cubic(pro_bn[pindex_0], pro_bn[pindex_1], pro_bn[pindex_2], pro_bn[pindex_3], pamount)
        sex_W = cubic(sex_Wn[pindex_0], sex_Wn[pindex_1], sex_Wn[pindex_2], sex_Wn[pindex_3], pamount)
        sex_b = cubic(sex_bn[pindex_0], sex_bn[pindex_1], sex_bn[pindex_2], sex_bn[pindex_3], pamount)
        str_W = cubic(str_Wn[pindex_0], str_Wn[pindex_1], str_Wn[pindex_2], str_Wn[pindex_3], pamount)
        str_b = cubic(str_bn[pindex_0], str_bn[pindex_1], str_bn[pindex_2], str_bn[pindex_3], pamount)
        
        W0.astype(np.float32).tofile('./Parameters/' + mname + '/W0_%03i.bin' % i)
        W1.astype(np.float32).tofile('./Parameters/' + mname + '/W1_%03i.bin' % i)
        W2.astype(np.float32).tofile('./Parameters/' + mname + '/W2_%03i.bin' % i)
        
        b0.astype(np.float32).tofile('./Parameters/' + mname + '/b0_%03i.bin' % i)
        b1.astype(np.float32).tofile('./Parameters/' + mname + '/b1_%03i.bin' % i)
        b2.astype(np.float32).tofile('./Parameters/' + mname + '/b2_%03i.bin' % i)

        ang_W.astype(np.float32).tofile('./Parameters/' + mname + '/ang_W_%03i.bin' % i)
        ang_b.astype(np.float32).tofile('./Parameters/' + mname + '/ang_b_%03i.bin' % i)
        chi_W.astype(np.float32).tofile('./Parameters/' + mname + '/chi_W_%03i.bin' % i)
        chi_b.astype(np.float32).tofile('./Parameters/' + mname + '/chi_b_%03i.bin' % i)
        dep_W.astype(np.float32).tofile('./Parameters/' + mname + '/dep_W_%03i.bin' % i)
        dep_b.astype(np.float32).tofile('./Parameters/' + mname + '/dep_b_%03i.bin' % i)
        neu_W.astype(np.float32).tofile('./Parameters/' + mname + '/neu_W_%03i.bin' % i)
        neu_b.astype(np.float32).tofile('./Parameters/' + mname + '/neu_b_%03i.bin' % i)
        old_W.astype(np.float32).tofile('./Parameters/' + mname + '/old_W_%03i.bin' % i)
        old_b.astype(np.float32).tofile('./Parameters/' + mname + '/old_b_%03i.bin' % i)
        pro_W.astype(np.float32).tofile('./Parameters/' + mname + '/pro_W_%03i.bin' % i)
        pro_b.astype(np.float32).tofile('./Parameters/' + mname + '/pro_b_%03i.bin' % i)
        sex_W.astype(np.float32).tofile('./Parameters/' + mname + '/sex_W_%03i.bin' % i)
        sex_b.astype(np.float32).tofile('./Parameters/' + mname + '/sex_b_%03i.bin' % i)
        str_W.astype(np.float32).tofile('./Parameters/' + mname + '/str_W_%03i.bin' % i)
        str_b.astype(np.float32).tofile('./Parameters/' + mname + '/str_b_%03i.bin' % i)
        
""" Construct Network """

network = PhaseFunctionedNetwork(rng=rng, input_shape=X.shape[1]+1, output_shape=Y.shape[1], dropout=0.7)

""" Construct Trainer """

batchsize = 32
epochs = 100
epochs = 25
# epochs = 1 # Use this to check all folders are set up correctly before committing to full training
trainer = AdamTrainer(rng=rng, batchsize=batchsize, epochs=epochs, alpha=0.0001)

""" Split data into groups of size batch_size and organise the data so that we cycle through the styles """
no_of_clips = np.sum(L_default, axis=0)[::w]   # no of non-mirrored clips in each style
no_of_batches = no_of_clips//(batchsize//2)  # each batch is half mirrored and half not
min_no_of_batches = int(min([np.floor(no_of_batches[i]) for i in xrange(len(no_of_batches)) if no_of_batches[i]!=0]))

""" Shuffle all data to avoid any problems caused by capturing similar motions together. """

assert len(X_default) == len(X_mirror)

I=np.arange(len(X_default))
rng.shuffle(I)
X_default=X_default[I]
Y_default=Y_default[I]
L_default=L_default[I]
P_default=P_default[I]
X_mirror=X_mirror[I]
Y_mirror=Y_mirror[I]
L_mirror=L_mirror[I]
P_mirror=P_mirror[I]

X_data_default = []
Y_data_default = []
L_data_default = []
P_data_default = []
X_data_mirror = []
Y_data_mirror = []
L_data_mirror = []
P_data_mirror = []
for i in xrange(len(no_of_batches)):
    if no_of_batches[i]==0: 
        pass
    else:
        indexes = np.where(L_default[:,w*i] == 1)[0]

        X_keep_default = X_default[indexes][:min_no_of_batches*(batchsize//2)]
        Y_keep_default = Y_default[indexes][:min_no_of_batches*(batchsize//2)]
        L_keep_default = L_default[indexes][:min_no_of_batches*(batchsize//2)]
        P_keep_default = P_default[indexes][:min_no_of_batches*(batchsize//2)]
        X_keep_mirror = X_mirror[indexes][:min_no_of_batches*(batchsize//2)]
        Y_keep_mirror = Y_mirror[indexes][:min_no_of_batches*(batchsize//2)]
        L_keep_mirror = L_mirror[indexes][:min_no_of_batches*(batchsize//2)]
        P_keep_mirror = P_mirror[indexes][:min_no_of_batches*(batchsize//2)]

        X_data_default.append(X_keep_default)
        Y_data_default.append(Y_keep_default)
        L_data_default.append(L_keep_default)
        P_data_default.append(P_keep_default)
        X_data_mirror.append(X_keep_mirror)
        Y_data_mirror.append(Y_keep_mirror)
        L_data_mirror.append(L_keep_mirror)
        P_data_mirror.append(P_keep_mirror)


X_train = []
Y_train = []
L_train = []
P_train = []
for i in xrange(min_no_of_batches):
    for j in xrange(len(X_data_default)):
        X_train.append(X_data_default[j][i*(batchsize//2):(i+1)*(batchsize//2)])
        X_train.append(X_data_mirror[j][i*(batchsize//2):(i+1)*(batchsize//2)])
        Y_train.append(Y_data_default[j][i*(batchsize//2):(i+1)*(batchsize//2)])
        Y_train.append(Y_data_mirror[j][i*(batchsize//2):(i+1)*(batchsize//2)])
        L_train.append(L_data_default[j][i*(batchsize//2):(i+1)*(batchsize//2)])
        L_train.append(L_data_mirror[j][i*(batchsize//2):(i+1)*(batchsize//2)])
        P_train.append(P_data_default[j][i*(batchsize//2):(i+1)*(batchsize//2)])
        P_train.append(P_data_mirror[j][i*(batchsize//2):(i+1)*(batchsize//2)])

X_in = np.concatenate(X_train, axis=0)
Y_in = np.concatenate(Y_train, axis=0)
L_in = np.concatenate(L_train, axis=0)
P_in = np.concatenate(P_train, axis=0)
print("After batch processing...")
print(X_in.shape, Y_in.shape, L_in.shape, P_in.shape)

""" Start Training """


start=time.time()      
E = theano.shared(np.concatenate([X_in, P_in[...,np.newaxis]], axis=-1), borrow=True)
F = theano.shared(Y_in, borrow=True)
G = theano.shared(L_in, borrow=True)
trainer.train(network, E, F, G, filename='./Parameters/' + mname + '/network.npz', restart=False, shuffle=False)
end=time.time()
elapsed = np.array([end-start]) 

""" Save Network """

save_network(network)
np.savez_compressed('./Training_Stats/' + mname + '_stats.npz', n_epochs=trainer.total_epochs[1:], train_loss=trainer.train_losses[1:], time=elapsed)


