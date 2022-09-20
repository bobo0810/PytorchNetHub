import numpy as np
from numba import cuda

@cuda.jit
def mm_stride(A, B, C):

    grid_column, grid_row = cuda.grid(2)
    stride_column, stride_row = cuda.gridsize(2)
    
    for data_row in range(grid_row, A.shape[0], stride_row):
        for data_column in range(grid_column, B.shape[1], stride_column):
            sum = 0
            for i in range(A.shape[1]): # `range(B.shape[0])` is also okay
                sum += A[data_row][i] * B[i][data_column]
                
            C[data_row][data_column] = sum

n = 1024
a = np.arange(n*n).reshape(n,n).astype(np.int32)
b = np.arange(n*n).reshape(n,n).astype(np.int32)
c = np.zeros((a.shape[0], b.shape[1])).astype(np.int32)

d_a = cuda.to_device(a)
d_b = cuda.to_device(b)
d_c = cuda.to_device(c)

ts = (32,32)
bs = (32,32)

mm_stride[bs, ts](d_a, d_b, d_c)