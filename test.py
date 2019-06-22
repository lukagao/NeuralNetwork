import numpy
import matplotlib.pyplot as plt
from matplotlib.image import imread
import conf

#x=numpy.arange(0,6,0.1)
#y=numpy.sin(x)
#plt.plot(x,y)
#plt.show()
#img=imread(conf.img_path)
#plt.imshow(img)
#plt.show()
a=numpy.array([[-2,3,4],[5,2,-1],[2,2,2]])
print(a>=0)
print(a[a>=0])
print(a[numpy.array([[False,True,True],[True,True,False],[True,True,True]])])