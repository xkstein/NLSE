import pyfftw
import mlx.core as mx
import scipy
import numpy as np

def split_step_mlx(A_0, t, Z, make_differential_operator: callable, make_nonlinear_operator: callable, n_integral_iterations = 10):
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
    dZ = Z[1] - Z[0]

    A = mx.zeros((Z.size, A_0.size), dtype=mx.complex64)
    A[0] = A_0

    if scipy.fft.next_fast_len(A_0.size) != A_0.size:
        print('using slow fft')
    omega = 2 * np.pi * np.fft.fftfreq(A_0.size, d=t[1] - t[0])
    D = make_differential_operator(omega)
    differential_operator = mx.exp( dZ / 2 * D )

    Af_start = mx.fft.fft(A[0])
    for ind, A_start in enumerate(A[:-1]):
        A_end = A_start[:]

        N_start = make_nonlinear_operator(A_start)
        DA_start = mx.fft.ifft( differential_operator * Af_start )

        mx.eval(DA_start)
        mx.eval(N_start)
        mx.eval(A_end)

        for refinement_ind in range(n_integral_iterations):
            if refinement_ind == 0:
                N_end = N_start
            else:
                N_end = mx.array( make_nonlinear_operator(A_end), dtype=mx.complex64 )
            A_mid = mx.exp( dZ / 2 * ( N_start + N_end ) ) * DA_start
            Af_end = differential_operator * mx.fft.fft(A_mid)
            A_end = mx.fft.ifft( Af_end )
            mx.eval(Af_end)
            mx.eval(A_end)

        Af_start = Af_end
        A[ind + 1] = A_end
    return A

def split_step_pyfftw(A_0, t, Z, make_differential_operator: callable, make_nonlinear_operator: callable, n_integral_iterations = 10):
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
    dZ = Z[1] - Z[0]

    A = np.zeros((Z.size, A_0.size), dtype=np.cdouble)
    A[0] = A_0

    if scipy.fft.next_fast_len(A_0.size) != A_0.size:
        print('using slow fft')
    omega = 2 * np.pi * scipy.fft.fftfreq(A_0.size, d=t[1] - t[0])
    D = make_differential_operator(omega)
    differential_operator = np.exp( dZ / 2 * D )

    A_start = pyfftw.empty_aligned(A_0.size, dtype='complex128')
    A_mid = pyfftw.empty_aligned(A_0.size, dtype='complex128')
    A_end = pyfftw.empty_aligned(A_0.size, dtype='complex128')
    Af_end = pyfftw.empty_aligned(A_0.size, dtype='complex128')

    A_start[:] = A_0
    A_end[:] = A_0
    Af_start = pyfftw.interfaces.numpy_fft.fft(A_0)
    for ind in range(len(A) - 1):
        #A_end[:] = A_start

        N_start = make_nonlinear_operator(A_start)
        DA_start = pyfftw.interfaces.numpy_fft.ifft( differential_operator * Af_start )

        for refinement_ind in range(n_integral_iterations):
            if refinement_ind == 0:
                N_end = N_start
            else:
                N_end = make_nonlinear_operator(A_end)
            A_mid[:] = np.exp( dZ / 2 * ( N_start + N_end ) ) * DA_start
            Af_end = differential_operator * pyfftw.interfaces.numpy_fft.fft(A_mid)
            A_end = pyfftw.interfaces.numpy_fft.ifft( Af_end )

        A_start[:] = A_end
        Af_start[:] = Af_end
        A[ind + 1] = A_end
    return A

