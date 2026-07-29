import scipy
import numpy as np

def split_step(A_0, t, Z, make_differential_operator: callable, make_nonlinear_operator: callable, n_integral_iterations = 10):
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

    omega = 2 * np.pi * np.fft.fftfreq(A_0.size, d=t[1] - t[0])
    D = make_differential_operator(omega)
    differential_operator = np.exp( dZ / 2 * D )

    Af_start = np.fft.fft(A_0)
    for ind, A_start in enumerate(A[:-1]):
        A_end = np.copy(A_start)

        N_start = make_nonlinear_operator(A_start)
        DA_start = np.fft.ifft( differential_operator * Af_start )

        for refinement_ind in range(n_integral_iterations):
            if refinement_ind == 0:
                N_end = N_start
            else:
                N_end = make_nonlinear_operator(A_end)
            A_mid = np.exp( dZ / 2 * ( N_start + N_end ) ) * DA_start
            Af_end = differential_operator * np.fft.fft(A_mid)
            A_end = np.fft.ifft( Af_end )

        Af_start = Af_end
        A[ind + 1] = A_end
    return A

def split_step_fast(A_0, t, Z, make_differential_operator: callable, make_nonlinear_operator: callable, n_integral_iterations = 10):
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

    Af_start = scipy.fft.fft(A_0)
    for ind, A_start in enumerate(A[:-1]):
        A_end = np.copy(A_start)

        N_start = make_nonlinear_operator(A_start)
        DA_start = scipy.fft.ifft( differential_operator * Af_start )

        for refinement_ind in range(n_integral_iterations):
            if refinement_ind == 0:
                N_end = N_start
            else:
                N_end = make_nonlinear_operator(A_end)
            A_mid = np.exp( dZ / 2 * ( N_start + N_end ) ) * DA_start
            Af_end = differential_operator * scipy.fft.fft(A_mid)
            A_end = scipy.fft.ifft( Af_end )

        Af_start = Af_end
        A[ind + 1] = A_end
    return A
