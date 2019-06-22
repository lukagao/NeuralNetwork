from network.testNet import TestNet
from network.trainNet import TrainNet
import conf

def loopTrain():
    learning_rate = 0.01
    while True:
        train = TrainNet(conf.input_nodes, conf.hidden_nodes, conf.output_nodes, learning_rate)
        net = train.do_train()
        # net = train.net
        accuracy = TestNet(net).do_test()
        print(learning_rate)
        print(accuracy)
        if accuracy >= 1.0:
            train.save_weights()
            print(learning_rate)
            break
        learning_rate += 0.01

def testWeight():
    train = TrainNet(conf.input_nodes, conf.hidden_nodes, conf.output_nodes, conf.learning_rate)
    net = train.net
    accuracy = TestNet(net).do_test()
    print(accuracy)


if __name__=='__main__':
    loopTrain()
    #testWeight()

