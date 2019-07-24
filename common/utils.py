import numpy

def img2col(img,filter_h,filter_w,stride=1,pad=0):
    N,C,H,W = img.shape
    out_h=(H+2*pad-filter_h)//stride +1
    out_w=(W+2*pad-filter_w)//stride +1

    img = numpy.pad(img,[(0,0),(0,0),(pad,pad),(pad,pad)],mode='constant')
    col = numpy.zeros((N,C,out_h,out_w,filter_h,filter_w))

    for h in range(out_h):
        h_max = h*stride + filter_h
        for w in range(out_w):
            w_max = w*stride + filter_w
            col[:,:,h,w,:,:] = img[:,:,h*stride:h_max:1,w*stride:w_max:1]

    col = col.transpose(0,2,3,1,4,5).reshape(N*out_h*out_w,-1)
    return col

def col2img(col,img_shape, filter_h, filter_w, stride=1, pad=0):
    N,C,H,W = img_shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    col = col.reshape(N,out_h,out_w,C,filter_h,filter_w).transpose(0,3,1,2,4,5)
    img = numpy.zeros((N,C,H+2*pad,W+2*pad))
    for h in range(out_h):
        h_max = h*stride +filter_h
        for w in range(out_w):
            w_max = w*stride + filter_w
            img[:,:,stride*h:h_max:1,stride*w:w_max:1] = col[:,:,h,w,:,:]
    return img[:,:,pad:H+pad,pad:W+pad]

def img2col_ex(img, filter_h, filter_w, stride=1, pad=0):
    N,C,H,W = img.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    img = numpy.pad(img,[(0,0),(0,0),(pad,pad),(pad,pad)],mode='constant')
    col = numpy.zeros((N,C,filter_h,filter_w,out_h,out_w))

    for h in range(filter_h):
        h_max = h + stride*out_h
        for w in range(filter_w):
            w_max = w +stride*out_w
            col[:,:,h,w,:,:] = img[:,:,h:h_max:stride,w:w_max:stride]
    col = col.transpose(0,4,5,1,2,3).reshape(N*out_h*out_w,-1)
    return col

def col2img_ex(col,img_shape, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = img_shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    col = col.reshape(N,out_h,out_w,C,filter_h,filter_w).transpose(0,3,4,5,1,2)
    img = numpy.zeros((N, C, H + 2 * pad, W + 2 * pad))
    for h in range(filter_h):
        h_max = h+out_h*stride
        for w in range(filter_w):
            w_max = w+out_w*stride
            img[:,:,h:h_max:stride,w:w_max:stride] = col[:,:,h,w,::]
    return img[:,:,pad:H+pad,pad:W+pad]



if __name__ == '__main__':
    img = numpy.random.normal(0.0, 1.0, (1, 3, 4, 4))
    print(img)
    col = img2col(img, 2, 2, 2, 0)
    print(col)
    img = col2img_ex(col,(1,3,4,4),2,2,2,0)
    print(img)
    #print(img2col_ex(a, 2, 2, 2, 0))



#a=numpy.array([[[1,2],[2,3]],[[1,2],[2,3]],[[1,2],[2,3]]])
#b=numpy.pad(a,[(0,0),(1,1),(1,2)],mode='constant')
#print(b)
#print(c)

