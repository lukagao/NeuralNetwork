import numpy
from scipy import special

class NeuralNetwork(object):

    #input,hidden,output nodes and learning rate
    def __init__(self,inodes,hnodes,onodes,lr):
        self.inodes=inodes
        self.hnodes=hnodes
        self.onodes=onodes
        self.lr=lr
        self.active_func=lambda x:special.expit(x)
        self.weights=self.init_weight()

    def init_weight(self):
        self.w_itoh=numpy.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
        self.w_htoo=numpy.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))
        l=[self.w_itoh,self.w_htoo]
        return l

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

#print(type(numpy.array(numpy.zeros(10))))
#l1=[1,2,3]
#l2=[1,2,3]
#a1=numpy.array(l1)
#a2=numpy.array(l2)
#print(a1*a2)