import conf
import numpy
from network.TrainCNN import TrainCNN


class MeasureCNN(object):

    def __init__(self,net,input_dim):
        self.net=net
        self.input_dim = input_dim
        self.testFile=open(conf.testData,'r')

    def measure(self):
        total=0.0
        correct=0.0
        while True:
            line = self.testFile.readline()
            if not line:
                break
            total += 1.0
            image_array = line.split(',')
            inputs = (numpy.asfarray(image_array[1:]) / 255.0 * 0.99) + 0.01
            #print('number is : ' + image_array[0])
            number=int(image_array[0])
            inputs = inputs.reshape(*self.input_dim)
            #print(self.net.predict(inputs))
            out=numpy.argmax(self.net.predict(inputs)[0])
            if number==out:
                correct+=1.0
            #print('net output is :'+str(out))
            #print('---------------------------------------------------------------------------------------')
        self.testFile.close()
        return correct/total

def loop_measure():
    while True:
        train = TrainCNN()
        net = train.do_train()
        # net = train.net
        # print('train finnish')
        measure = MeasureCNN(net, train.input_dim)
        print('**************************************************')
        print(conf.learning_rate)
        print(measure.measure())
        print('**************************************************')
        conf.learning_rate+=0.01

if __name__ == '__main__':
    train = TrainCNN()
    net = train.do_train()
    # net = train.net
    print('train finnish')
    measure = MeasureCNN(net,train.input_dim)
    print('start test')
    print(measure.measure())
    #loop_measure()
