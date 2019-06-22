import conf
from network.neural import NeuralNetwork
import numpy
import aircv



class TrainNet(object):
    def __init__(self,input_nodes,hidden_nodes,output_nodes,lr):
        self.output_nodes=output_nodes
        self.net=NeuralNetwork(input_nodes,hidden_nodes,output_nodes,lr)

    def do_train(self):
        with open(conf.trainData, 'r') as trainFile:
            while True:
                line = trainFile.readline()
                if not line:
                    break
                image_array = line.split(',')
                inputs = (numpy.asfarray(image_array[1:]) / 255.0 * 0.99) + 0.01
                targets = numpy.zeros(self.output_nodes) + 0.01
                value = int(image_array[0])
                targets[value] = 0.99
                self.net.train(inputs, targets)
        #self.save_weights(list(self.net.weights.values()))
        return self.net

    def save_weights(self):
        weights=list(self.net.weights.values())
        self.do_save(conf.weights,weights[0],'w')
        for index in range(1,len(weights)):
            self.do_save(conf.weights,weights[index],'a')

    def do_save(self,path,weights,mode):
        with open(path, mode) as f:
            w_list = weights.flatten()
            size=len(w_list)
            for index in range(size):
                f.write(str(w_list[index]))
                if index < (size-1):
                    f.write(',')
            f.write('\n')




