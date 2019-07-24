import numpy
from scipy import special
from common.utils import img2col,col2img


class CNNLayer(object):

    def __init__(self,w,b,stride=1,pad=0):
        self.w = w
        self.b = b
        self.stride = stride
        self.pad = pad
        #权重偏置该变量
        self.dw = None
        self.db = None
        #输入图片数据以及转换后的矩阵
        self.x = None
        self.col = None
        self.col_w = None

    def forward(self,x):
        FN, FC, FH, FW = self.w.shape
        N, C, H, W =x.shape
        out_h = (H + 2*self.pad - FH)//self.stride +1
        out_w = (W + 2*self.pad - FW)//self.stride +1
        col = img2col(x, FH, FW, self.stride,self.pad)
        col_w = self.w.reshape(FN, -1).T
        out = numpy.dot(col,col_w) + self.b
        out = out.reshape(N, out_h,out_w,-1).transpose(0,3,1,2)
        self.x = x
        self.col = col
        self.col_w = col_w
        return out

    def backward(self,do):
        FN, FC, FH, FW = self.w.shape
        do = do.transpose(0,2,3,1).reshape(-1,FN)
        self.db = numpy.sum(do,axis=0)
        self.dw = numpy.dot(self.col.T,self.do)
        self.dw = self.dw.transpose(1,0).reshape(FN, FC, FH, FW)
        dcol = numpy.dot(do,self.col_w.T)
        dx = col2img(dcol,self.x.shape,FH,FW,self.stride,self.pad)
        return dx




class SoftMaxWithLossLayer(object):
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self,x,t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y,self.t)

    def backward(self,do=1):
        batch_size = self.y.shape[0]
        return (self.y-self.t)*do/batch_size

class ReLULayer(object):

    def __init__(self):
        #掩码：标识小于0的数据的位置
        self.mask = None

    def forward(self,x):
        self.mask = (x<=0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self,dout):
        dout[self.mask] = 0
        dx = dout
        return dx

class AffineLayer(object):

    def __init__(self,w):
        self.w=w

    def forward(self,x):
        self.x=x
        return numpy.dot(x,self.w)

    def backward(self,do):
        dx=numpy.dot(do,self.w.T)
        self.dw=numpy.dot(self.x.T,do)
        return dx


class SigmoidLayer(object):

    def __init__(self):
        self.out=None

    def forward(self,x):
        self.out=special.expit(x)
        return self.out

    def backward(self,do):
        return do*(1.0-self.out)*self.out

#平方差误差层
class SquareErrorLayer(object):

    def __init__(self):
        self.out = None
        self.y = None
        self.t = None

    def forward(self,y,t):
        self.y = y
        self.t = t
        self.out=(t-y)**2
        return self.out

    def backward(self,do=1):
        return (self.y-self.t)*do

#交叉熵误差函数
def cross_entropy_error(y,t):
    if y.ndim ==1:
        y = y.reshape(1,y.size)
        t = t.reshape(1,t.size)
    batch_size = y.shape[0]
    return -numpy.sum(t*numpy.log(y + 1e-7))/batch_size

#SoftMax函数
def softmax(a):
    exp_a = numpy.exp(a)
    return exp_a/numpy.sum(exp_a)


a = numpy.array([[[1,1],[2,2],[2,1]]])
print(softmax(a))

