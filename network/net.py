import numpy

class NeuralNetwork(object):

    #input,hidden,output nodes and learning rate
    def __init__(self,inodes,hnodes,onodes,lr):
        self.inodes=inodes
        self.hnodes=hnodes
        self.onodes=onodes
        self.lr=lr
        self.init_weight()

    def init_weight(self):
        self.w_itoh=numpy.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
        self.w_htoo=numpy.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))

    def train(self):
        pass

    def query(self):
        pass

inodes=3
hnodes=3
onodes=3
lr=0.5
net=NeuralNetwork(inodes,hnodes,onodes,lr)
print(net.w_htoo)
print(net.w_itoh)