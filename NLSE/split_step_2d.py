import numpy as np
from typing import Any, Union
from numpy.typing import NDArray

numeric = Union[int, float, complex, np.number]

def split_step_2d(A_0: NDArray[np.number], t: NDArray[np.number], 
                  Z: NDArray[np.number], T_0: numeric, beta_2: numeric, 
                  diff_beta: numeric, gamma: numeric, 
                  n_integral_iterations=3) -> NDArray[Any]:
    '''Finds evolution of pulse A_0 as it propagates along Z
    A_0: Array containing the pulse entering the fiber (A_X, A_Y). Should be 
            shape (2, t.size)
    t:   Local time at each step
    Z:   Z position of each step in solver
    T_0: Pulse parameter (expecting A_0 to be in a form like P * sech(t/T_0))
    diff_beta: Birefringence in the fiber -> (Beta_1_x - Beta_1_y)

    This is expecting to recieve and return X and Y polarized light

    The variables are in the notation in Agarwal's "Nonlinear Fiber Optics"
    '''
    intensity = np.abs( A_0[0] ) ** 2 + np.abs( A_0[1] ) ** 2
    P_0 = np.max( intensity )
    L_D = T_0 ** 2 / np.abs( beta_2 )
    L_NL = 1 / ( gamma * P_0 )

    N = np.sqrt( L_D / L_NL )

    tau = t / T_0
    Xi = Z / L_D

    u = N * A_0[0] / np.sqrt( P_0 )
    v = N * A_0[1] / np.sqrt( P_0 )
    LCP = ( u + 1j * v ) / np.sqrt( 2 )
    RCP = ( u - 1j * v ) / np.sqrt( 2 )

    delta = diff_beta * T_0 / ( 2 * beta_2 )
    
    A = split_step_2d_unitless(np.array((LCP, RCP)), tau, Xi, delta, n_integral_iterations=n_integral_iterations)

    out = np.zeros(A.shape, dtype=complex)

    LCP_out = A[:,0]
    RCP_out = A[:,1]

    out[:,0] = ( LCP_out + RCP_out ) / np.sqrt( 2 )
    out[:,1] = -1j * ( LCP_out - RCP_out ) / np.sqrt( 2 )

    return np.sqrt( P_0 ) * out / N

def split_step_2d_unitless(A_0: NDArray[np.number], tau: NDArray[np.number], 
                           Xi: NDArray[np.number], delta: numeric, 
                           n_integral_iterations=3) -> NDArray[Any]:
    '''Finds evolution of pulse A_0 as it propagates along Z
    A_0: Array containing the pulse entering the fiber (LCP, RCP). Should be 
            shape (2, t.size)
    delta: Unitless birefringence -> (Beta_1x - Beta_1y) T_0 / ( 2 | beta_2 | )

    This operates with LCP and RCP light, using "unitless" formulation

    The variables are in the notation in Agarwal's "Nonlinear Fiber Optics"
    '''
    dtau = tau[1] - tau[0]
    dXi = Xi[1] - Xi[0]

    A = np.zeros((Xi.size, 2, A_0[0].size), dtype=complex)
    A[0,0] = A_0[0]
    A[0,1] = A_0[1]

    # Define D operators
    f = np.fft.fftfreq(tau.size, d=dtau)
    omega = 2 * np.pi * f

    D_x = -1j * delta * omega - 1j / 2 * omega ** 2
    D_y =  1j * delta * omega - 1j / 2 * omega ** 2

    use_D = lambda A, D: np.fft.ifft( np.exp(D * dXi / 2) * np.fft.fft(A) )
    make_N = lambda A_same, A_opp: 1j * ( np.abs(A_same) ** 2 + 2 * np.abs(A_opp) ** 2)

    for ind, (LCP, RCP) in enumerate(A[:-1]):
        LCP_at_h = np.copy(LCP)
        RCP_at_h = np.copy(RCP)

        DLCP = use_D(LCP, D_x)
        DRCP = use_D(RCP, D_y)

        NL_at_0 = make_N(LCP, RCP)
        NR_at_0 = make_N(RCP, LCP)

        for _ in range(n_integral_iterations + 1):
            NL_at_h = make_N(LCP_at_h, RCP_at_h)
            NR_at_h = make_N(RCP_at_h, LCP_at_h)

            LCP_at_h= use_D(
                        np.exp( dXi / 2 * ( NL_at_h + NL_at_0 ) ) * DLCP, 
                        D_x
                    )
            RCP_at_h = use_D(
                        np.exp( dXi / 2 * ( NR_at_h + NR_at_0 ) ) * DRCP, 
                        D_y
                    )

        A[ind + 1,0] = LCP_at_h
        A[ind + 1,1] = RCP_at_h

    return A
