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
        self.save_weights(self.net.w_itoh,self.net.w_htoo)
        return self.net

    def save_weights(self,w_itoh,w_htoo):
        self.do_save(conf.weights,w_itoh,'w')
        self.do_save(conf.weights,w_htoo,'a')

    def do_save(self,path,weights,mode):
        with open(path, mode) as f:
            w_list = weights.flatten()
            size=len(w_list)
            for index in range(size):
                f.write(str(w_list[index]))
                if index < (size-1):
                    f.write(',')
            f.write('\n')




