@cuda.jit
def add_matrix_stride(A, B, C):

    y, x = cuda.grid(2)
    stride_y, stride_x = cuda.gridsize(2)
    
    for i in range(x, A.shape[0], stride_x):
        for j in range(y, A.shape[1], stride_y):
            C[i][j] = A[i][j] + B[i][j]

A = np.arange(64*64).reshape(64, 64).astype(np.int32)
B = A * 2
C = np.zeros_like(A)
d_A = cuda.to_device(A)
d_B = cuda.to_device(B)
d_C = cuda.to_device(C)

blocks = (6,6)
threads_per_block = (6,6)

add_matrix_stride[blocks, threads_per_block](d_A, d_B, d_C)