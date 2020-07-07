from os import path
trainData_100='D:\\DevelopTool\\Workspace\\Git\\AIData\\mnist\\train\\mnist_train_100.csv'
testData_10='D:\\DevelopTool\\Workspace\\Git\\AIData\\mnist\\test\\mnist_test_10.csv'

trainData='D:\\DevelopTool\\Workspace\\Git\\AIData\\mnist\\train\\mnist_train.csv'
testData='D:\\DevelopTool\\Workspace\\Git\\AIData\\mnist\\test\\mnist_test.csv'
input_nodes=784
hidden_nodes=100
output_nodes=10
learning_rate=0.001
weights=path.join(path.abspath(path.dirname(__file__)),'weights.csv')
img_path='D:\\DevelopTool\\Workspace\\Git\\AIData\\image\\ssh.png'
#0.1 97
#0.01 9392
#0.05 9798
#0.06 9822
#0.065 9814
#0.062 9803
#0.07 9808
#0.08 9789