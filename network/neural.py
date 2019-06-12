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

    def train(self):
        pass

    def query(self,inputs):
        #initial inputs is initial outpus
        outputs=inputs
        for weight in self.weights:
            inputs=numpy.dot(weight,outputs)
            outputs=self.active_func(inputs)
        return outputs




inodes=3
hnodes=3
onodes=3
lr=0.5
net=NeuralNetwork(inodes,hnodes,onodes,lr)
print(net.query([1,2,3]))