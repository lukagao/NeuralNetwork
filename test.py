import numpy
import matplotlib.pyplot as plt
from matplotlib.image import imread
import conf

x=numpy.arange(0,6,0.1)
y=numpy.sin(x)
#plt.plot(x,y)
#plt.show()
img=imread(conf.img_path)
plt.imshow(img)
plt.show()