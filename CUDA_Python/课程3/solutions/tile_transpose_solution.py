@cuda.jit
def tile_transpose(a, transposed):
    # `tile_transpose` assumes it is launched with a 32x32 block dimension,
    # and that `a` is a multiple of these dimensions.
    
    # 1) Create 32x32 shared memory array.
    tile = cuda.shared.array((32, 32), numba_types.int32)
    
    # Compute offsets into global input array. Recall for coalesced access we want to map threadIdx.x increments to
    # the fastest changing index in the data, i.e. the column in our array.
    a_col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    a_row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    # 2) Make coalesced read from global memory into shared memory array.
    # Note the use of local thread indices for the shared memory write,
    # and global offsets for global memory read.
    tile[cuda.threadIdx.y, cuda.threadIdx.x] = a[a_row, a_col]

    # 3) Wait for all threads in the block to finish updating shared memory.
    cuda.syncthreads()

    # 4) Calculate transposed location for the shared memory array tile
    # to be written back to global memory. Note that blockIdx.y*blockDim.y 
    # and blockIdx.x* blockDim.x are swapped (because we want to write to the
    # transpose locations), but we want to keep access coalesced, so match up the
    # threadIdx.x to the fastest changing index, i.e. the column
    t_col = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.x
    t_row = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.y

    # 5) Write from shared memory (using thread indices)
    # back to global memory (using grid indices)
    # transposing each element within the shared memory array.
    transposed[t_row, t_col] = tile[cuda.threadIdx.x, cuda.threadIdx.y]