@cuda.jit
def col_sums(a, sums, ds):
    idx = cuda.grid(1)
    sum = 0.0

    for i in range(ds):
        sum += a[i][idx]

    sums[idx] = sum
