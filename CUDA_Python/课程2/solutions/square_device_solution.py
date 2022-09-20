import numpy as np
from numba import cuda

@cuda.jit
def square_device(a, out):
    idx = cuda.grid(1)
    out[idx] = a[idx]**2
    
n = 4096
a = np.arange(n)

d_a = cuda.to_device(a)
d_out = cuda.device_array(shape=(n,), dtype=np.float32)

threads = 32
blocks = 128

square_device[blocks, threads](d_a, d_out)