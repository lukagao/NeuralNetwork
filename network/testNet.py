from network.trainNet import TrainNet
import conf
import numpy
class TestNet(object):

    def __init__(self,net):
        self.net=net
        self.testFile=open(conf.testData_10,'r')

    def do_test(self):
        while True:
            line = self.testFile.readline()
            if not line:
                break
            image_array = line.split(',')
            inputs = (numpy.asfarray(image_array[1:]) / 255.0 * 0.99) + 0.01
            print('number is : ' + image_array[0])
            out=numpy.argmax(self.net.query(inputs))
            print('net output is :'+str(out))
            print('---------------------------------------------------------------------------------------')
        self.testFile.close()
