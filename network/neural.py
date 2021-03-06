import numpy
from os import path
import conf
from collections import OrderedDict
from network.Layers import SquareErrorLayer,SigmoidLayer,AffineLayer


class NeuralNetwork(object):

    #input,hidden,output nodes and learning rate
    def __init__(self,inodes,hnodes,onodes,lr):
        self.inodes=inodes
        self.hnodes=hnodes
        self.onodes=onodes
        self.lr=lr
        self.w_keys=['Affine1','Affine2']
        self.weights = self.init_weight()
        self.b = self.init_b()
        self.layers=self.init_layers()
        self.lastLayer=SquareErrorLayer()

    def init_weight(self):
        weights=OrderedDict()
        if path.exists(conf.weights):
            w1,w2=self.load_weights()
        else:
            w1=numpy.random.normal(0.0,pow(self.hnodes,-0.5),(self.inodes,self.hnodes))
            w2=numpy.random.normal(0.0,pow(self.onodes,-0.5),(self.hnodes,self.onodes))
        weights['Affine1']=w1
        weights['Affine2']=w2
        return weights

    def init_b(self):
        b = OrderedDict()
        b['b1'] = numpy.zeros(self.hnodes)
        b['b2'] = numpy.zeros(self.onodes)
        return b


    def load_weights(self):
        print('use exist weights')
        with open(conf.weights, 'r') as f:
            l = f.readlines()
            w1 = numpy.asfarray(l[0].split(',')).reshape(self.inodes,self.hnodes)
            w2 = numpy.asfarray(l[1].split(',')).reshape(self.hnodes,self.onodes)
        return w1,w2

    def init_layers(self):
        layers=OrderedDict()
        layers['Affine1']=AffineLayer(self.weights['Affine1'],self.b['b1'])
        layers['Sigmoid1']=SigmoidLayer()
        layers['Affine2'] = AffineLayer(self.weights['Affine2'],self.b['b2'])
        layers['Sigmoid2'] = SigmoidLayer()
        return layers

    def error(self,inputs,targets):
        inputs=self.query(inputs)
        self.lastLayer.forward(inputs,targets)
        #return self.lastLayer.backward(inputs,targets)
        return self.lastLayer.backward(do=1)

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
            if isinstance(layer,AffineLayer):
                layer.w-=self.lr*layer.dw
        for key in self.w_keys:
            self.weights[key]=self.layers[key].w




#l1=[1,2,3]
#l2=[1,2,3]
#a1=numpy.array(l1)
#a2=numpy.array(l2)
#print(a1*a2)