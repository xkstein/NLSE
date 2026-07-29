import scipy
import numpy as np
import jax.numpy as jnp
import jax

def split_step_single_jax(A_0, t, dZ, make_differential_operator: callable, make_nonlinear_operator: callable, n_integral_iterations = 10):
    '''
    So we find `N(z+h)` by running through the calculation assuming `N(z+h) = N(z)` to find `A(z+h)` to plug back in to find `N(z+h)`

    THIS CODE DOES NOT DO:
    $$e^{\frac{h}{2}D}[ e^{\frac{h}{2}N}e^{\frac{h}{2}D}A(z, T) e^{\frac{h}{2}N}e^{\frac{h}{2}D}A(z + h, T) ]$$

    It does
    $$e^{\frac{h}{2}D} e^{\frac{h}{2} ( N(A(z)) + N(A(z+h)) )} e^{\frac{h}{2}D} A(z)$$

    ```python
    N_start = N(A(z))
    N_end = N(A(z+h))
    DA_start = exp(D * h / 2) A(z, T)
    A_mid = exp(( N(A(z)) + N(A(z+h)) ) * h / 2) exp(D * h / 2) A(z, T)
    ```

    We have to fft n_integral_iterations times per step. You would think its 
    n_integral_iterations + 1, but you can save Af from the last iteration.
    '''

    omega = 2 * jnp.pi * jnp.fft.fftfreq(A_0.size, d=t[1] - t[0])
    D = make_differential_operator(omega)
    differential_operator = jnp.exp( dZ / 2 * D )
    
    A_start = jnp.array(A_0, dtype=jnp.complex64)
    Af_start = jnp.fft.fft(A_start)
    A_end = jnp.copy(A_start)

    N_start = make_nonlinear_operator(A_start)
    DA_start = jnp.fft.ifft( differential_operator * Af_start )

    def refinement(ind, N_end):
        A_mid = jnp.exp( dZ / 2 * ( N_start + N_end ) ) * DA_start
        Af_end = differential_operator * jnp.fft.fft(A_mid)
        A_end = jnp.fft.ifft( Af_end )
        return make_nonlinear_operator( A_end )

    N_end = jax.lax.fori_loop(0, n_integral_iterations - 1, refinement, N_start)
    A_mid = jnp.exp( dZ / 2 * ( N_start + N_end ) ) * DA_start
    Af_end = differential_operator * jnp.fft.fft(A_mid)
    A_end = jnp.fft.ifft( Af_end )
    return A_end