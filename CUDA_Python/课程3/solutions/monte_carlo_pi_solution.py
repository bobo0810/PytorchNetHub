@cuda.jit
def monte_carlo_pi_device(rng_states, nsamples, out):
    thread_id = cuda.grid(1)

    # Compute pi by drawing random (x, y) points and finding what
    # fraction lie inside a unit circle
    acc = 0
    for i in range(nsamples):
        x = xoroshiro128p_uniform_float32(rng_states, thread_id)
        y = xoroshiro128p_uniform_float32(rng_states, thread_id)
        if x**2 + y**2 <= 1.0:
            acc += 1

    out[thread_id] = 4.0 * acc / nsamples
    
nsamples = 10000000
threads_per_block = 128
blocks = 32
grid_size = threads_per_block * blocks

samples_per_thread = int(nsamples / grid_size)
rng_states = create_xoroshiro128p_states(grid_size, seed=1)
d_out = cuda.device_array(threads_per_block * blocks, dtype=np.float32)

monte_carlo_pi_device[blocks, threads_per_block](rng_states, samples_per_thread, d_out)