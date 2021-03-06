import numpy
from scipy import special
from os import path
import conf

class NeuralNetwork_Old(object):

    #input,hidden,output nodes and learning rate
    def __init__(self,inodes,hnodes,onodes,lr):
        self.inodes=inodes
        self.hnodes=hnodes
        self.onodes=onodes
        self.lr=lr
        self.active_func=lambda x:special.expit(x)
        self.weights=self.init_weight()

    def init_weight(self):
        if path.exists(conf.weights):
            self.w_itoh,self.w_htoo=self.load_weights()
        else:
            self.w_itoh=numpy.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
            self.w_htoo=numpy.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))
        l=[self.w_itoh,self.w_htoo]
        return l

    def load_weights(self):
        print('use exist weights')
        with open(conf.weights, 'r') as f:
            l = f.readlines()
            w_itoh = numpy.asfarray(l[0].split(',')).reshape(self.hnodes,self.inodes)
            w_htoo = numpy.asfarray(l[1].split(',')).reshape(self.onodes,self.hnodes)
        return w_itoh,w_htoo

    def train(self,inputs,targets):
        targets=numpy.array(targets,ndmin=2).T
        inputs=numpy.array(inputs,ndmin=2).T
        #get hidden layer inputs and outputs
        hidden_i=numpy.dot(self.w_itoh,inputs)
        hidden_o=self.active_func(hidden_i)
        #get output layse inputs and outputs
        output_i=numpy.dot(self.w_htoo,hidden_o)
        outputs=self.active_func(output_i)
        #get output layer errors and hidden layers errors
        o_e=targets-outputs
        h_e=numpy.dot(self.w_htoo.T,o_e)
        #get htoo weight changes
        w_htoo_c=self.lr*numpy.dot(o_e*outputs*(1.0-outputs),numpy.transpose(hidden_o))
        #get itoh weight changes
        w_itoh_c=self.lr*numpy.dot(h_e*hidden_o*(1-hidden_o),numpy.transpose(inputs))

        self.w_htoo+=w_htoo_c
        self.w_itoh+=w_itoh_c
        #w_htoo_c)
        #print(w_itoh_c)


    def query(self,inputs):
        #initial inputs is initial outpus
        outputs=numpy.array(inputs,ndmin=2).T
        for weight in self.weights:
            inputs=numpy.dot(weight,outputs)
            outputs=self.active_func(inputs)
        return outputs

