from network.testNet import TestNet
from network.trainNet import TrainNet
import conf

if __name__=='__main__':
    train = TrainNet(conf.input_nodes,conf.hidden_nodes,conf.output_nodes,conf.learning_rate)
    net = train.do_train()
    #net = train.net
    TestNet(net).do_test()