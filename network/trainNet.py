import conf
from network.neural import NeuralNetwork
import numpy
import aircv



class TrainNet(object):
    def __init__(self,input_nodes,hidden_nodes,output_nodes,lr):
        self.output_nodes=output_nodes
        self.net=NeuralNetwork(input_nodes,hidden_nodes,output_nodes,lr)
        self.trainFile=open(conf.trainData,'r')


    def do_train(self):
        while True:
            line = self.trainFile.readline()
            if not line:
                break
            image_array = line.split(',')
            inputs = (numpy.asfarray(image_array[1:]) / 255.0 * 0.99) + 0.01
            targets = numpy.zeros(self.output_nodes) + 0.01
            value = int(image_array[0])
            targets[value] = 0.99
            self.net.train(inputs, targets)
        self.trainFile.close()
        return self.net
