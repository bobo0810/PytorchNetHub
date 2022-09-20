@cuda.jit
def add_matrix(A, B, C):
    i,j = cuda.grid(2)
    
    C[j,i] = A[j,i] + B[j,i]
    
A = np.arange(36*36).reshape(36, 36).astype(np.int32)
B = A * 2
C = np.zeros_like(A)
d_A = cuda.to_device(A)
d_B = cuda.to_device(B)
d_C = cuda.to_device(C)

blocks = (6,6)
threads_per_block = (6,6)

add_matrix[blocks, threads_per_block](d_A, d_B, d_C)