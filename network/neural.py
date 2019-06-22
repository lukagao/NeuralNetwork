import numpy
from scipy import special
from os import path
import conf
from collections import OrderedDict

class AffineLayer(object):

    def __init__(self,w):
        self.w=w

    def forward(self,x):
        self.x=x
        return numpy.dot(x,self.w)

    def backward(self,do):
        dx=numpy.dot(do,self.w.T)
        self.dw=numpy.dot(self.x.T,do)
        self.w-=self.dw
        return dx


class SigmoidLayer(object):

    def __init__(self):
        self.out=None

    def forward(self,x):
        self.out=special.expit(x)
        return self.out

    def backward(self,do):
        return do*(1.0-self.out)*self.out

class SquareErrorLayer(object):

    def __init__(self):
        self.out=None

    def forward(self,x,t):
        self.out=(t-x)**2
        return self.out

    def backward(self,x,t):
        return x-t


class NeuralNetwork(object):

    #input,hidden,output nodes and learning rate
    def __init__(self,inodes,hnodes,onodes,lr):
        self.inodes=inodes
        self.hnodes=hnodes
        self.onodes=onodes
        self.lr=lr
        self.w_keys=['w1','w2']
        self.weights = self.init_weight()
        self.layers=self.init_layers()
        self.lastLayer=SquareErrorLayer()

    def init_weight(self):
        weights={}
        if path.exists(conf.weights):
            w1,w2=self.load_weights()
        else:
            w1=numpy.random.normal(0.0,pow(self.hnodes,-0.5),(self.inodes,self.hnodes))
            w2=numpy.random.normal(0.0,pow(self.onodes,-0.5),(self.hnodes,self.onodes))
        weights['w1']=w1
        weights['w2']=w2
        return weights

    def load_weights(self):
        print('use exist weights')
        with open(conf.weights, 'r') as f:
            l = f.readlines()
            w1 = numpy.asfarray(l[0].split(',')).reshape(self.inodes,self.hnodes)
            w2 = numpy.asfarray(l[1].split(',')).reshape(self.hnodes,self.onodes)
        return w1,w2

    def init_layers(self):
        layers=OrderedDict()
        layers['Affine1']=AffineLayer(self.weights['w1'])
        layers['Sigmoid1']=SigmoidLayer()
        layers['Affine2'] = AffineLayer(self.weights['w2'])
        layers['Sigmoid2'] = SigmoidLayer()
        return layers

    def error(self,inputs,targets):
        inputs=self.query(inputs)
        return self.lastLayer.backward(inputs,targets)

    def query(self,inputs):
        #initial inputs is initial outpus
        inputs=numpy.array(inputs,ndmin=2)
        for layer in self.layers.values():
            inputs=layer.forward(inputs)
        return inputs

    def train(self,inputs,targets):
        do=self.error(inputs,targets)
        layers=list(self.layers.values())
        layers.reverse()
        for layer in layers:
            do=layer.backward(do)



#l1=[1,2,3]
#l2=[1,2,3]
#a1=numpy.array(l1)
#a2=numpy.array(l2)
#print(a1*a2)