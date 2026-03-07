import numpy as np
import pytest

from NLSE import split_step

tau = np.linspace(-80, 80, 4096)
xi_max = 10 * np.pi / 2
beta2_sign = -1

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

def nl_operator(u):
    return 1j * np.abs(u) ** 2

def diff_operator(omega):
    return 1j * beta2_sign / 2 * omega ** 2

def find_error(N, steps):
    u_gt = get_ground_truth(N)

    xi = np.linspace(0, xi_max, steps)

    u = split_step(
        u_gt(0),
        tau,
        xi,
        diff_operator,
        nl_operator,
    )

    e = u[-1] - u_gt(xi[-1])

    return L2_norm(tau, e) / L2_norm(tau, u_gt(xi[-1]))

@pytest.mark.parametrize("N", [1, 2])
def test_split_step_accuracy(N):
    error = find_error(N, 4096)

    # adjust tolerance depending on solver accuracy
    assert error < 1e-2
