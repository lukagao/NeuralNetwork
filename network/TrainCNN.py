import conf
from network.ConvNeural import ConvNeural
import numpy
import aircv



class TrainCNN(object):
    def __init__(self,input_dim=(1,1,28,28),filter_num=30,filter_w=5,filter_h=5,pad=0,
                 stride=1,hidden_size=100,output_size=10,w_std=0.01):
        self.input_dim = input_dim
        self.output_size = output_size
        self.net=ConvNeural(input_dim=input_dim,filter_num=filter_num,filter_w=filter_w,filter_h=filter_h,pad=pad,
                 stride=stride,hidden_size=hidden_size,output_size=output_size,w_std=w_std)

    def do_train(self):
        with open(conf.trainData, 'r') as trainFile:
            while True:
                line = trainFile.readline()
                if not line:
                    break
                image_array = line.split(',')
                inputs = (numpy.asfarray(image_array[1:]) / 255.0 * 0.99) + 0.01
                inputs = inputs.reshape(*self.input_dim)
                # print('train_input:')
                # print(inputs)
                targets = numpy.zeros(self.output_size) + 0.01
                value = int(image_array[0])
                targets[value] = 0.99
                targets = targets.reshape(1,-1)
                # print('target is: '+ str(value))
                self.net.train(inputs, targets)
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


if __name__ == '__main__':
    train = TrainCNN()
    train.do_train()


