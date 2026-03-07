import numpy as np
import pytest

from NLSE import split_step

tau = np.linspace(-80, 80, 4096)
xi_max = 10 * np.pi / 2

def L2_norm(x, y):
    return np.sqrt(np.trapezoid(np.abs(y) ** 2, x))

def get_ground_truth(N):
    if N == 1:
        return lambda xi: 1 / np.cosh(tau) * np.exp(1j * xi / 2)

    elif N == 2:
        return lambda xi: (
            4
            * np.exp(1j * xi / 2)
            * (
                np.cosh(3 * tau)
                + 3 * np.exp(4j * xi) * np.cosh(tau)
            )
            / (
                np.cosh(4 * tau)
                + 4 * np.cosh(2 * tau)
                + 3 * np.cos(4 * xi)
            )
        )

    else:
        raise NotImplementedError

def find_error(N, steps):
    u_gt = get_ground_truth(N)

    beta2 = -4.20056799728266e-27 # s^2 m^-1
    T_0 = 1e-12

    A_eff = 50e-12 # in m^2
    n_2 = 3.2e-20 # m^2 W^-1
    lambda_0 = 1.55e-6
    gamma = 2 * np.pi * n_2 / (lambda_0 * A_eff)
    
    L_D = T_0 ** 2 / abs( beta2 )
    P_0 = N ** 2 / ( L_D * gamma )
    L_NL = L_D / ( N ** 2 )
    
    def nl_operator(A):
        return 1j * gamma * np.abs(A) ** 2
    
    def diff_operator(omega):
        return 1j * beta2 / 2 * omega ** 2
    
    t = tau * T_0
    z_max = xi_max * L_D
    z = np.linspace(0, z_max, steps)
    
    A_0 = u_gt(0) * np.sqrt(P_0) / N
    A = split_step(A_0, t, z, diff_operator, nl_operator, n_integral_iterations=10)

    A_ref = u_gt(xi_max) * np.sqrt(P_0) / N
    e = A[-1] - A_ref
    return L2_norm(tau, e) / L2_norm(tau, A_ref)

@pytest.mark.parametrize("N", [1, 2])
def test_split_step_accuracy(N):
    error = find_error(N, 4096)
    print(error)

    # adjust tolerance depending on solver accuracy
    assert error < 1e-2
