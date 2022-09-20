# Add your solution here
@cuda.jit
def cuda_histogram(x, xmin, xmax, histogram_out):
    '''Increment bin counts in histogram_out, given histogram range [xmin, xmax).'''
    nbins = histogram_out.shape[0] # 分为N组
    bin_width = (xmax - xmin) / nbins # 每组宽度
    
    
    start = cuda.grid(1)
    
    stride=cuda.gridsize(1) # 1指 所有进程按一维下标索引
    for i in range(start,x.shape[0],stride):
        bin_number=(x[i] - xmin)/bin_width # 所有进程的一次并行计算
        if bin_number >= 0 and bin_number < histogram_out.shape[0]:
            cuda.atomic.add(histogram_out, bin_number, 1)# 原子操作 全局加1