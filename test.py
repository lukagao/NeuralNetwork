import numpy
import matplotlib.pyplot as plt
from matplotlib.image import imread
import conf
from collections import OrderedDict

#x=numpy.arange(0,6,0.1)
#y=numpy.sin(x)
#plt.plot(x,y)
#plt.show()
#img=imread(conf.img_path)
#plt.imshow(img)
#plt.show()
#a=numpy.array([[-2,3,4],[5,2,-1],[2,2,2]])
#print(a>=0)
#print(a[a>=0])
#print(a[numpy.array([[False,True,True],[True,True,False],[True,True,True]])])

# o=OrderedDict()
# o['1']=[1]
# o['2']=[2]
# o['3']=[3]
#
# l=list(o.values())
# l[0].append(4)
# l.reverse()
# print(o)
# print(l)

class A(object):
    pass

a = A()
b= a
print(a)
del a
print(b)
print(a)
