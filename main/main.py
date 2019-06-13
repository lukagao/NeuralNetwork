from network.testNet import TestNet
from network.trainNet import TrainNet
import conf

if __name__=='__main__':
    net=TrainNet(conf.input_nodes,conf.hidden_nodes,conf.output_nodes,conf.learning_rate).do_train()
    TestNet(net).do_test()