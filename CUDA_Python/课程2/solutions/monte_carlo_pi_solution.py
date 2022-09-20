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
