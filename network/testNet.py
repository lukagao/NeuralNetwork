import conf
import numpy
class TestNet(object):

    def __init__(self,net):
        self.net=net
        self.testFile=open(conf.testData,'r')

    def do_test(self):
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
            out=numpy.argmax(self.net.query(inputs))
            if number==out:
                correct+=1.0
            #print('net output is :'+str(out))
            #print('---------------------------------------------------------------------------------------')
        self.testFile.close()
        return correct/total
