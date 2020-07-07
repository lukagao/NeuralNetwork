from network.Layers import *
import numpy
from collections import OrderedDict
import conf

class ConvNeural(object):

    def __init__(self,input_dim=(1,1,28,28),filter_num=30,filter_w=5,filter_h=5,pad=0,
                 stride=1,hidden_size=100,output_size=10,w_std=0.01):
        N,C,H,W = input_dim
        conv_output_h = (W-filter_h+2*pad)//stride + 1
        conv_output_w = (H-filter_w+2*pad)//stride + 1
        # N*pool_output_size 为pool层输出总大小
        pool_output_size = (conv_output_h//2)*(conv_output_w//2)*filter_num
        self.params = dict()
        self.params['conv_pool_w'] = numpy.random.randn(filter_num,C,filter_h,filter_w)*w_std
        self.params['conv_pool_b'] = numpy.zeros(filter_num)
        self.params['pool_hidden_w'] = numpy.random.randn(pool_output_size,hidden_size)*w_std
        self.params['pool_hidden_b'] = numpy.zeros(hidden_size)
        self.params['hidden_output_w'] = numpy.random.randn(hidden_size,output_size)*w_std
        self.params['hidden_output_b'] = numpy.zeros(output_size)

        self.layers = OrderedDict()
        self.layers['conv'] = CNNLayer(self.params['conv_pool_w'],b=self.params['conv_pool_b'],stride=stride,pad=pad)
        self.layers['relu1'] = ReLULayer()
        self.layers['pool'] = PoolingLayer(2,2,stride=2)
        self.layers['affine1'] = AffineLayer(self.params['pool_hidden_w'],b=self.params['pool_hidden_b'])
        self.layers['relu2'] = ReLULayer()
        self.layers['affine2'] = AffineLayer(self.params['hidden_output_w'],b=self.params['hidden_output_b'])
        # self.last_layer = SoftMaxWithLossLayer()
        self.layers['Sigmoid1'] = SigmoidLayer()
        self.last_layer = SquareErrorLayer()


    def predict(self,x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self,x,t):
        x = self.predict(x)
        return self.last_layer.forward(x,t)

    def train(self,x,t):

        self.loss(x,t)
        dout = self.last_layer.backward(do=1)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
            #更新权重：
            if type(layer) == CNNLayer or type(layer) == AffineLayer:
                # print(layer.w)
                layer.w -= layer.dw*conf.learning_rate
                layer.b -= layer.db*conf.learning_rate






