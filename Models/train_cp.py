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
from DiagLayer import DiagLayer

""" 
Training the main pfnn with residual adapters decomposed using CP decomposition and with the central diagonal tensor of size 30x30.
"""

mname='CP'
rng = np.random.RandomState(23456)

""" Load Data """

database = np.load('./Data/style_database.npz')
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

X = np.concatenate((X_default, X_mirror), axis=0)
Y = np.concatenate((Y_default, Y_mirror), axis=0)
P = np.concatenate((P_default, P_mirror), axis=0)
L = np.concatenate((L_default, L_mirror), axis=0)

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


        self.ang_W0 = HiddenLayer((1, 30, 512), rng=rng, gamma=0.01)
        self.chi_W0 = HiddenLayer((1, 30, 512), rng=rng, gamma=0.01)
        self.dep_W0 = HiddenLayer((1, 30, 512), rng=rng, gamma=0.01)
        self.neu_W0 = HiddenLayer((1, 30, 512), rng=rng, gamma=0.01)
        self.old_W0 = HiddenLayer((1, 30, 512), rng=rng, gamma=0.01)
        self.pro_W0 = HiddenLayer((1, 30, 512), rng=rng, gamma=0.01)
        self.sex_W0 = HiddenLayer((1, 30, 512), rng=rng, gamma=0.01)
        self.str_W0 = HiddenLayer((1, 30, 512), rng=rng, gamma=0.01)

        self.ang_W1 = DiagLayer((self.nslices, 1, 30), rng=rng, gamma=0.01)
        self.chi_W1 = DiagLayer((self.nslices, 1, 30), rng=rng, gamma=0.01)
        self.dep_W1 = DiagLayer((self.nslices, 1, 30), rng=rng, gamma=0.01)
        self.neu_W1 = DiagLayer((self.nslices, 1, 30), rng=rng, gamma=0.01)
        self.old_W1 = DiagLayer((self.nslices, 1, 30), rng=rng, gamma=0.01)
        self.pro_W1 = DiagLayer((self.nslices, 1, 30), rng=rng, gamma=0.01)
        self.sex_W1 = DiagLayer((self.nslices, 1, 30), rng=rng, gamma=0.01)
        self.str_W1 = DiagLayer((self.nslices, 1, 30), rng=rng, gamma=0.01)

        self.ang_W2 = HiddenLayer((1, 512, 30), rng=rng, gamma=0.01)
        self.chi_W2 = HiddenLayer((1, 512, 30), rng=rng, gamma=0.01)
        self.dep_W2 = HiddenLayer((1, 512, 30), rng=rng, gamma=0.01)
        self.neu_W2 = HiddenLayer((1, 512, 30), rng=rng, gamma=0.01)
        self.old_W2 = HiddenLayer((1, 512, 30), rng=rng, gamma=0.01)
        self.pro_W2 = HiddenLayer((1, 512, 30), rng=rng, gamma=0.01)
        self.sex_W2 = HiddenLayer((1, 512, 30), rng=rng, gamma=0.01)
        self.str_W2 = HiddenLayer((1, 512, 30), rng=rng, gamma=0.01)

        self.ang_b = BiasLayer((1, 512))
        self.chi_b = BiasLayer((1, 512))
        self.dep_b = BiasLayer((1, 512))
        self.neu_b = BiasLayer((1, 512))
        self.old_b = BiasLayer((1, 512))
        self.pro_b = BiasLayer((1, 512))
        self.sex_b = BiasLayer((1, 512))
        self.str_b = BiasLayer((1, 512))

        self.layers = [
            self.W0, self.W1, self.W2,
            self.b0, self.b1, self.b2,
            self.ang_W0, self.ang_W1,
            self.ang_W2, self.ang_b,
            self.chi_W0, self.chi_W1,
            self.chi_W2, self.chi_b,
            self.dep_W0, self.dep_W1,
            self.dep_W2, self.dep_b,
            self.neu_W0, self.neu_W1,
            self.neu_W2, self.neu_b,
            self.old_W0, self.old_W1,
            self.old_W2, self.old_b,
            self.pro_W0, self.pro_W1,
            self.pro_W2, self.pro_b,
            self.sex_W0, self.sex_W1,
            self.sex_W2, self.sex_b,
            self.str_W0, self.str_W1,
            self.str_W2, self.str_b]

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

        ang_W1 = cubic(self.ang_W1.W[pindex_0], self.ang_W1.W[pindex_1], self.ang_W1.W[pindex_2], self.ang_W1.W[pindex_3], Wamount)
        chi_W1 = cubic(self.chi_W1.W[pindex_0], self.chi_W1.W[pindex_1], self.chi_W1.W[pindex_2], self.chi_W1.W[pindex_3], Wamount)
        dep_W1 = cubic(self.dep_W1.W[pindex_0], self.dep_W1.W[pindex_1], self.dep_W1.W[pindex_2], self.dep_W1.W[pindex_3], Wamount)
        neu_W1 = cubic(self.neu_W1.W[pindex_0], self.neu_W1.W[pindex_1], self.neu_W1.W[pindex_2], self.neu_W1.W[pindex_3], Wamount)
        old_W1 = cubic(self.old_W1.W[pindex_0], self.old_W1.W[pindex_1], self.old_W1.W[pindex_2], self.old_W1.W[pindex_3], Wamount)
        pro_W1 = cubic(self.pro_W1.W[pindex_0], self.pro_W1.W[pindex_1], self.pro_W1.W[pindex_2], self.pro_W1.W[pindex_3], Wamount)
        sex_W1 = cubic(self.sex_W1.W[pindex_0], self.sex_W1.W[pindex_1], self.sex_W1.W[pindex_2], self.sex_W1.W[pindex_3], Wamount)
        str_W1 = cubic(self.str_W1.W[pindex_0], self.str_W1.W[pindex_1], self.str_W1.W[pindex_2], self.str_W1.W[pindex_3], Wamount)
        
        sty_index = T.cast(theano.shared(np.zeros((32,))), 'int32') # Takes the same matrix for every phase in the batch
        ang_W0 = self.ang_W0.W[sty_index]
        chi_W0 = self.chi_W0.W[sty_index]
        dep_W0 = self.dep_W0.W[sty_index]
        neu_W0 = self.neu_W0.W[sty_index]
        old_W0 = self.old_W0.W[sty_index]
        pro_W0 = self.pro_W0.W[sty_index]
        sex_W0 = self.sex_W0.W[sty_index]
        str_W0 = self.str_W0.W[sty_index]

        ang_W2 = self.ang_W2.W[sty_index]
        chi_W2 = self.chi_W2.W[sty_index]
        dep_W2 = self.dep_W2.W[sty_index]
        neu_W2 = self.neu_W2.W[sty_index]
        old_W2 = self.old_W2.W[sty_index]
        pro_W2 = self.pro_W2.W[sty_index]
        sex_W2 = self.sex_W2.W[sty_index]
        str_W2 = self.str_W2.W[sty_index]

        ang_b = self.ang_b.b[sty_index]
        chi_b = self.chi_b.b[sty_index]
        dep_b = self.dep_b.b[sty_index]
        neu_b = self.neu_b.b[sty_index]
        old_b = self.old_b.b[sty_index]
        pro_b = self.pro_b.b[sty_index]
        sex_b = self.sex_b.b[sty_index]
        str_b = self.str_b.b[sty_index]

        H0 = input[:,:-1]
        H1 = self.activation(T.batched_dot(W0, self.dropout0(H0)) + b0)

        ang_H3 = T.switch(T.all(T.eq(label[0],self.ang_label)), T.batched_dot(W2, self.dropout2(self.activation(T.batched_dot(W1, self.dropout1(H1)) + b1 + T.batched_dot(ang_W2, (ang_W1.reshape([32,30]) * T.batched_dot(ang_W0, self.dropout1(H1)))) + ang_b))) + b2, self.zeros)
        chi_H3 = T.switch(T.all(T.eq(label[0],self.chi_label)), T.batched_dot(W2, self.dropout2(self.activation(T.batched_dot(W1, self.dropout1(H1)) + b1 + T.batched_dot(chi_W2, (chi_W1.reshape([32,30]) * T.batched_dot(chi_W0, self.dropout1(H1)))) + chi_b))) + b2, self.zeros)
        dep_H3 = T.switch(T.all(T.eq(label[0],self.dep_label)), T.batched_dot(W2, self.dropout2(self.activation(T.batched_dot(W1, self.dropout1(H1)) + b1 + T.batched_dot(dep_W2, (dep_W1.reshape([32,30]) * T.batched_dot(dep_W0, self.dropout1(H1)))) + dep_b))) + b2, self.zeros)
        neu_H3 = T.switch(T.all(T.eq(label[0],self.neu_label)), T.batched_dot(W2, self.dropout2(self.activation(T.batched_dot(W1, self.dropout1(H1)) + b1 + T.batched_dot(neu_W2, (neu_W1.reshape([32,30]) * T.batched_dot(neu_W0, self.dropout1(H1)))) + neu_b))) + b2, self.zeros)
        old_H3 = T.switch(T.all(T.eq(label[0],self.old_label)), T.batched_dot(W2, self.dropout2(self.activation(T.batched_dot(W1, self.dropout1(H1)) + b1 + T.batched_dot(old_W2, (old_W1.reshape([32,30]) * T.batched_dot(old_W0, self.dropout1(H1)))) + old_b))) + b2, self.zeros)
        pro_H3 = T.switch(T.all(T.eq(label[0],self.pro_label)), T.batched_dot(W2, self.dropout2(self.activation(T.batched_dot(W1, self.dropout1(H1)) + b1 + T.batched_dot(pro_W2, (pro_W1.reshape([32,30]) * T.batched_dot(pro_W0, self.dropout1(H1)))) + pro_b))) + b2, self.zeros)
        sex_H3 = T.switch(T.all(T.eq(label[0],self.sex_label)), T.batched_dot(W2, self.dropout2(self.activation(T.batched_dot(W1, self.dropout1(H1)) + b1 + T.batched_dot(sex_W2, (sex_W1.reshape([32,30]) * T.batched_dot(sex_W0, self.dropout1(H1)))) + sex_b))) + b2, self.zeros)
        str_H3 = T.switch(T.all(T.eq(label[0],self.str_label)), T.batched_dot(W2, self.dropout2(self.activation(T.batched_dot(W1, self.dropout1(H1)) + b1 + T.batched_dot(str_W2, (str_W1.reshape([32,30]) * T.batched_dot(str_W0, self.dropout1(H1)))) + str_b))) + b2, self.zeros)

        
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

    ang_W0 = network.ang_W0.W.get_value()
    chi_W0 = network.chi_W0.W.get_value()
    dep_W0 = network.dep_W0.W.get_value()
    neu_W0 = network.neu_W0.W.get_value()
    old_W0 = network.old_W0.W.get_value()
    pro_W0 = network.pro_W0.W.get_value()
    sex_W0 = network.sex_W0.W.get_value()
    str_W0 = network.str_W0.W.get_value()

    ang_W1n = network.ang_W1.W.get_value()
    chi_W1n = network.chi_W1.W.get_value()
    dep_W1n = network.dep_W1.W.get_value()
    neu_W1n = network.neu_W1.W.get_value()
    old_W1n = network.old_W1.W.get_value()
    pro_W1n = network.pro_W1.W.get_value()
    sex_W1n = network.sex_W1.W.get_value()
    str_W1n = network.str_W1.W.get_value()

    ang_W2 = network.ang_W2.W.get_value()
    chi_W2 = network.chi_W2.W.get_value()
    dep_W2 = network.dep_W2.W.get_value()
    neu_W2 = network.neu_W2.W.get_value()
    old_W2 = network.old_W2.W.get_value()
    pro_W2 = network.pro_W2.W.get_value()
    sex_W2 = network.sex_W2.W.get_value()
    str_W2 = network.str_W2.W.get_value()

    ang_b = network.ang_b.b.get_value()
    chi_b = network.chi_b.b.get_value()
    dep_b = network.dep_b.b.get_value()
    neu_b = network.neu_b.b.get_value()
    old_b = network.old_b.b.get_value()
    pro_b = network.pro_b.b.get_value()
    sex_b = network.sex_b.b.get_value()
    str_b = network.str_b.b.get_value()
    
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
        
        W0.astype(np.float32).tofile('./Parameters/' + mname + '/W0_%03i.bin' % i)
        W1.astype(np.float32).tofile('./Parameters/' + mname + '/W1_%03i.bin' % i)
        W2.astype(np.float32).tofile('./Parameters/' + mname + '/W2_%03i.bin' % i)
        
        b0.astype(np.float32).tofile('./Parameters/' + mname + '/b0_%03i.bin' % i)
        b1.astype(np.float32).tofile('./Parameters/' + mname + '/b1_%03i.bin' % i)
        b2.astype(np.float32).tofile('./Parameters/' + mname + '/b2_%03i.bin' % i)

        ang_W1 = cubic(ang_W1n[pindex_0], ang_W1n[pindex_1], ang_W1n[pindex_2], ang_W1n[pindex_3], pamount)
        chi_W1 = cubic(chi_W1n[pindex_0], chi_W1n[pindex_1], chi_W1n[pindex_2], chi_W1n[pindex_3], pamount)
        dep_W1 = cubic(dep_W1n[pindex_0], dep_W1n[pindex_1], dep_W1n[pindex_2], dep_W1n[pindex_3], pamount)
        neu_W1 = cubic(neu_W1n[pindex_0], neu_W1n[pindex_1], neu_W1n[pindex_2], neu_W1n[pindex_3], pamount)
        old_W1 = cubic(old_W1n[pindex_0], old_W1n[pindex_1], old_W1n[pindex_2], old_W1n[pindex_3], pamount)
        pro_W1 = cubic(pro_W1n[pindex_0], pro_W1n[pindex_1], pro_W1n[pindex_2], pro_W1n[pindex_3], pamount)
        sex_W1 = cubic(sex_W1n[pindex_0], sex_W1n[pindex_1], sex_W1n[pindex_2], sex_W1n[pindex_3], pamount)
        str_W1 = cubic(str_W1n[pindex_0], str_W1n[pindex_1], str_W1n[pindex_2], str_W1n[pindex_3], pamount)

        ang_W1.astype(np.float32).tofile('./Parameters/' + mname + '/ang_W1_%03i.bin' % i)
        chi_W1.astype(np.float32).tofile('./Parameters/' + mname + '/chi_W1_%03i.bin' % i)
        dep_W1.astype(np.float32).tofile('./Parameters/' + mname + '/dep_W1_%03i.bin' % i)
        neu_W1.astype(np.float32).tofile('./Parameters/' + mname + '/neu_W1_%03i.bin' % i)
        old_W1.astype(np.float32).tofile('./Parameters/' + mname + '/old_W1_%03i.bin' % i)
        pro_W1.astype(np.float32).tofile('./Parameters/' + mname + '/pro_W1_%03i.bin' % i)
        sex_W1.astype(np.float32).tofile('./Parameters/' + mname + '/sex_W1_%03i.bin' % i)
        str_W1.astype(np.float32).tofile('./Parameters/' + mname + '/str_W1_%03i.bin' % i)

    ang_W0[0].astype(np.float32).tofile('./Parameters/' + mname + '/ang_W0.bin')
    chi_W0[0].astype(np.float32).tofile('./Parameters/' + mname + '/chi_W0.bin')
    dep_W0[0].astype(np.float32).tofile('./Parameters/' + mname + '/dep_W0.bin')
    neu_W0[0].astype(np.float32).tofile('./Parameters/' + mname + '/neu_W0.bin')
    old_W0[0].astype(np.float32).tofile('./Parameters/' + mname + '/old_W0.bin')
    pro_W0[0].astype(np.float32).tofile('./Parameters/' + mname + '/pro_W0.bin')
    sex_W0[0].astype(np.float32).tofile('./Parameters/' + mname + '/sex_W0.bin')
    str_W0[0].astype(np.float32).tofile('./Parameters/' + mname + '/str_W0.bin')

    ang_W2[0].astype(np.float32).tofile('./Parameters/' + mname + '/ang_W2.bin')
    chi_W2[0].astype(np.float32).tofile('./Parameters/' + mname + '/chi_W2.bin')
    dep_W2[0].astype(np.float32).tofile('./Parameters/' + mname + '/dep_W2.bin')
    neu_W2[0].astype(np.float32).tofile('./Parameters/' + mname + '/neu_W2.bin')
    old_W2[0].astype(np.float32).tofile('./Parameters/' + mname + '/old_W2.bin')
    pro_W2[0].astype(np.float32).tofile('./Parameters/' + mname + '/pro_W2.bin')
    sex_W2[0].astype(np.float32).tofile('./Parameters/' + mname + '/sex_W2.bin')
    str_W2[0].astype(np.float32).tofile('./Parameters/' + mname + '/str_W2.bin')

    ang_b[0].astype(np.float32).tofile('./Parameters/' + mname + '/ang_b.bin')
    chi_b[0].astype(np.float32).tofile('./Parameters/' + mname + '/chi_b.bin')
    dep_b[0].astype(np.float32).tofile('./Parameters/' + mname + '/dep_b.bin')
    neu_b[0].astype(np.float32).tofile('./Parameters/' + mname + '/neu_b.bin')
    old_b[0].astype(np.float32).tofile('./Parameters/' + mname + '/old_b.bin')
    pro_b[0].astype(np.float32).tofile('./Parameters/' + mname + '/pro_b.bin')
    sex_b[0].astype(np.float32).tofile('./Parameters/' + mname + '/sex_b.bin')
    str_b[0].astype(np.float32).tofile('./Parameters/' + mname + '/str_b.bin')
        
""" Construct Network """

network = PhaseFunctionedNetwork(rng=rng, input_shape=X.shape[1]+1, output_shape=Y.shape[1], dropout=0.7)

""" Construct Trainer """

batchsize = 32
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


