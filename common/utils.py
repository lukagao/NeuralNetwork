import numpy

def im2col(img,filter_h,filter_w,stride=1,pad=0):
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

def im2col_ex(img, filter_h, filter_w, stride=1, pad=0):
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



'''
def im2col_ex(input_data, filter_h, filter_w, stride=1, pad=0):

    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    img = numpy.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = numpy.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col

'''

if __name__ == '__main__':
    a = numpy.random.normal(0.0, 1.0, (1, 3, 4, 4))
    print(im2col(a, 2, 2, 2, 0))
    print(im2col_ex(a, 2, 2, 2, 0))



#a=numpy.array([[[1,2],[2,3]],[[1,2],[2,3]],[[1,2],[2,3]]])
#b=numpy.pad(a,[(0,0),(1,1),(1,2)],mode='constant')
#print(b)
#print(c)

