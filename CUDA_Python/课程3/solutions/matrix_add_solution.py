@cuda.jit
def matrix_add(a, b, out, coalesced):
    x, y = cuda.grid(2)
    
    if coalesced == True:
        out[y][x] = a[y][x] + b[y][x]
    else:
        out[x][y] = a[x][y] + b[x][y]
