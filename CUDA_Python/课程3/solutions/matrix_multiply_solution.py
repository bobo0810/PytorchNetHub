import numpy as np
from numba import cuda

@cuda.jit
def mm(a, b, c):
    column, row = cuda.grid(2)
    sum = 0
    
    for i in range(a.shape[0]):
        sum += a[row][i] * b[i][column]
        
    c[row][column] = sum
    
a = np.arange(16).reshape(4,4).astype(np.int32)
b = np.arange(16).reshape(4,4).astype(np.int32)
c = np.zeros_like(a)

d_a = cuda.to_device(a)
d_b = cuda.to_device(b)
d_c = cuda.to_device(c)

grid = (2,2)
block = (2,2)
mm[grid, block](d_a, d_b, d_c)