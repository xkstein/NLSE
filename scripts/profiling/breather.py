from NLSE import split_step_fast, split_step_pyfftw, split_step_mlx
from scipy.fft import  next_fast_len
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-noplot', action='store_true')
args = parser.parse_args()

beta2 = -4.20056799728266e-27 # s^2 m^-1

A_eff = 50e-12 # in m^2
n_2 = 3.2e-20 # m^2 W^-1
lambda_0 = 1.55e-6
gamma = 2 * np.pi * n_2 / (lambda_0 * A_eff)

N = 2
T_0 = 1e-12
FWHM = 1.76 * T_0
L_D = T_0 ** 2 / np.abs( beta2 )
P_0 = N ** 2 / ( L_D * gamma )
print(f'This simulation is happening for\n\tFWHM: {FWHM}s\n\tPower: {P_0} W\n\tN: {N}')

L_NL = 1 / ( gamma * P_0 )
assert np.isclose(L_D / L_NL, N ** 2), 'N is not self consistent!'

if args.noplot:
    t = np.linspace(-80e-12, 80e-12, next_fast_len(50000))
    Z = np.linspace(0, 10*np.pi/2 * L_D, 1024)
else:
    print('For plotting')
    t = np.linspace(-80e-12, 80e-12, 4096)
    Z = np.linspace(0, 10*np.pi/2 * L_D, 4096)

A_0 = np.sqrt(P_0) / np.cosh(t / T_0)

n_integral_iterations = 10
print(f'This travels to:\n\tZ: {Z[-1]:.2f} m\n\tXi: {Z[-1] / L_D}')
dZ = Z[1] - Z[0]

def nl_operator(A):
    #return 1j * gamma * np.abs(A) ** 2
    return 1j * gamma * ( A.conj() * A ).real

def diff_operator(omega):
    return 1j * beta2 / 2 * omega ** 2

A = split_step_mlx(A_0, t, Z, diff_operator, nl_operator)

if not args.noplot:
    import matplotlib.pyplot as plt

    mid = A[0].size // 2
    plt.figure(figsize=(7,7))
    plt.imshow(np.abs(A[::10,mid-200:mid+200])**2)
    plt.ylabel('Z')
    plt.xlabel('t')
    plt.show()
