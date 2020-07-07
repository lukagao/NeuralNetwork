import numpy
from scipy import special
from common.utils import img2col,col2img


class PoolingLayer(object):

    def __init__(self,pool_h,pool_w,stride=2,pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
        self.argmax = None
        self.x = None

    def forward(self,x):
        self.x = x
        N,C,H,W = x.shape
        out_h = (H - self.pool_h)//self.stride +1
        out_w = (W - self.pool_w)//self.stride +1
        col = img2col(x,self.pool_h,self.pool_w,self.stride,self.pad)
        col = col.reshape(-1,self.pool_h*self.pool_w)
        self.argmax = numpy.argmax(col,axis=1)
        out = numpy.max(col,axis=1)
        out = out.reshape(N,out_h,out_w,C).transpose(0,3,1,2)
        # print('pool out:')
        # print(out)
        return out

    def backward(self,do):
        do = do.transpose(0,2,3,1)
        pool_size = self.pool_h*self.pool_w
        dmax = numpy.zeros((do.size,pool_size))
        dmax[numpy.arange(self.argmax.size),self.argmax.flatten()] = do.flatten()
        dmax = dmax.reshape(do.shape+(pool_size,))
        dcol = dmax.reshape(dmax.shape[0]*dmax.shape[1]*dmax.shape[2],-1)
        dx = col2img(dcol,self.x.shape,self.pool_h,self.pool_w,self.stride,self.pad)
        return dx

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
        # print('cnn w:')
        # print(self.w)
        return out

    def backward(self,do):
        FN, FC, FH, FW = self.w.shape
        do = do.transpose(0,2,3,1).reshape(-1,FN)
        self.db = numpy.sum(do,axis=0)
        self.dw = numpy.dot(self.col.T,do)
        self.dw = self.dw.transpose(1,0).reshape(FN, FC, FH, FW)
        dcol = numpy.dot(do,self.col_w.T)
        dx = col2img(dcol,self.x.shape,FH,FW,self.stride,self.pad)
        return dx


class ReLULayer(object):

    def __init__(self):
        #掩码：标识小于0的数据的位置
        self.mask = None

    def forward(self,x):
        self.mask = (x<=0)
        out = x.copy()
        out[self.mask] = 0
        # print('relu out')
        # print(out)
        return out

    def backward(self,dout):
        dout[self.mask] = 0
        dx = dout
        return dx

class AffineLayer(object):

    def __init__(self,w,b):
        self.w = w
        self.b = b
        self.origin_shape = None

    def forward(self,x):
        self.x=x
        self.origin_shape = self.x.shape
        self.x = self.x.reshape(x.shape[0], -1)
        out = numpy.dot(self.x,self.w)+ self.b
        # print('affine out ')
        # print(out)
        return out

    def backward(self,do):
        dx=numpy.dot(do,self.w.T)
        self.dw=numpy.dot(self.x.T,do)
        self.db = numpy.sum(do, axis=0)

        dx = dx.reshape(*self.origin_shape)
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

class SoftMaxWithLossLayer(object):
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self,x,t):
        self.t = t
        print('soft input:')
        print(x)
        print('soft output')
        self.y = softmax(x)
        print(self.y)
        self.loss = cross_entropy_error(self.y,self.t)
        return self.loss

    def backward(self,do=1):
        batch_size = self.y.shape[0]
        return (self.y-self.t)*do/batch_size

#交叉熵误差函数
def cross_entropy_error(y,t):
    if y.ndim ==1:
        y = y.reshape(1,y.size)
        t = t.reshape(1,t.size)
    batch_size = y.shape[0]
    return -numpy.sum(t*numpy.log(y + 1e-7))/batch_size

#SoftMax函数
def softmax(a):
    if a.ndim == 2:
        a = a.T
        a = a - numpy.max(a, axis=0)
        y = numpy.exp(a) / numpy.sum(numpy.exp(a), axis=0)
        return y.T
    a = a - numpy.max(a)  # 溢出对策
    return numpy.exp(a) / numpy.sum(numpy.exp(a))


if __name__ == '__main__':
    a = numpy.array([[[1,1],[2,2],[2,1]]])
    print(a.shape)
    print(a.shape+(1,))

