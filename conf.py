from os import path
trainData_100='D:\\DevelopTool\\Workspace\\Git\\AIData\\mnist\\train\\mnist_train_100.csv'
testData_10='D:\\DevelopTool\\Workspace\\Git\\AIData\\mnist\\test\\mnist_test_10.csv'

trainData='D:\\DevelopTool\\Workspace\\Git\\AIData\\mnist\\train\\mnist_train.csv'
testData='D:\\DevelopTool\\Workspace\\Git\\AIData\\mnist\\test\\mnist_test.csv'
input_nodes=784
hidden_nodes=100
output_nodes=10
learning_rate=0.3
weights=path.join(path.abspath(path.dirname(__file__)),'weights.csv')
img_path='D:\\DevelopTool\\Workspace\\Git\\AIData\\image\\ssh.png'