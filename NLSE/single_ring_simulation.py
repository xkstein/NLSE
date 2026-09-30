from .split_step_1d_jax import split_step_single_jax
import jax
import jax.numpy as jnp

c = 299792458
hbar = 1.054571817e-34 # J s

class SingleRing:
    def __init__(self, *, neff, FSR, coupling_ring_bus, coupling_ring_drop, 
                 gamma, alpha_int, beta_2, omega_0, vacuum_noise=None, n_small_time=512):
        self.neff = neff,
        self.FSR = FSR
        self.coupling_ring_bus = coupling_ring_bus
        self.coupling_ring_drop = coupling_ring_drop
        self.gamma = gamma
        self.alpha_int = alpha_int
        self.alpha = (coupling_ring_bus + alpha_int) / 2
        self.beta_2 = beta_2
        self.omega_0 = omega_0

        if vacuum_noise is None:
            self.vacuum_noise = jnp.sqrt(hbar * omega_0 * FSR)

        _t_rt = 1 / FSR
        #self.time = jnp.linspace(-_t_rt / 2, _t_rt / 2, n_small_time)
        self.time = jnp.linspace(-_t_rt / 2, _t_rt / 2, n_small_time, endpoint=False)
        self.sim_omega = 2 * jnp.pi * jnp.fft.fftfreq(self.time.size, d=self.time[1] - self.time[0])

        self.frequency = jnp.fft.fftfreq(n_small_time, self.time[1] - self.time[0]) + omega_0 / ( 2 * jnp.pi )
        self.wavelength = c / self.frequency

        self.cavity_length = c / ( neff * FSR )

        self.time_normalized = jnp.sqrt( 2 * self.alpha / (jnp.abs(beta_2) * self.cavity_length) ) * self.time
        self.Iin_2_S = lambda Iin: jnp.sqrt( coupling_ring_bus * gamma * self.cavity_length / self.alpha ** 3 ) * jnp.sqrt(Iin)
        self.S_2_Iin = lambda S: jnp.abs( S ) ** 2 / jnp.abs( coupling_ring_bus * gamma * self.cavity_length / self.alpha ** 3 )

    def make_seed(self, detuning, intensity_in):
        '''
        This includes the pump field 

        Equation is direct from Yun's thesis equation 1.89. 

        Definition of the normalized units is from Breather paper, but it's the 
        same as Yun's thesis just without confusing fourier transform terms

        Arguments:
            detuning: real pump detuning
            intensity_in: pump intensity in W

        Returns:
            A_seed: Units are W^1/2 (not normalized field)
        '''
        S = jnp.sqrt(self.gamma * self.coupling_ring_bus * self.cavity_length / self.alpha ** 3) \
                * jnp.sqrt(intensity_in)
        F = self.make_seed_normalized(detuning / self.alpha, S)
        return F / jnp.sqrt(self.gamma * self.cavity_length / self.alpha)

    def make_seed_normalized(self, delta, S):
        '''
        This includes the pump field 

        Equation is direct from Yun's thesis equation 1.89. 

        Definition of the normalized units is from Breather paper, but it's the 
        same as Yun's thesis just without confusing fourier transform terms

        Arguments:
            delta: Normalized detuning
            S: Normalized pump field

        Returns:
            F: normalized field
        '''
        F = jnp.sqrt(2 * delta) * 1 / jnp.cosh(jnp.sqrt(delta) * self.time_normalized) \
                * jnp.exp(1j * jnp.arccos(jnp.sqrt(8 * delta) / (jnp.pi * S))) \
                - 1j * S / delta 
        return F


    def get_ikeda_map(self):
        ring_bus_bs = jnp.array([[jnp.sqrt(1 - self.coupling_ring_bus), 1j * jnp.sqrt(self.coupling_ring_bus)], 
                                [1j * jnp.sqrt(self.coupling_ring_bus), jnp.sqrt(1 - self.coupling_ring_bus)]])

        ring_drop_bs = jnp.array([[jnp.sqrt(1 - self.coupling_ring_drop), 1j * jnp.sqrt(self.coupling_ring_drop)], 
                                [1j * jnp.sqrt(self.coupling_ring_drop), jnp.sqrt(1 - self.coupling_ring_drop)]])

        def run_sim(A_init_main, A_in, detuning, nrt, n_per_step=2):
            '''
            Returns:
                Field in main ring
                Field coupled out into the bus waveguide
                Field coupled out into the drop waveguide
            '''
            _key_init = jax.random.key(0)
            _a_init_main = jnp.array(A_init_main, dtype=jnp.complex64)

            _a_in = jnp.array(A_in, dtype=jnp.complex64)
            _a_drop_in = jnp.zeros(A_in.shape, dtype=jnp.complex64)

            def nl_operator_main(A):
                return 1j * self.gamma * jnp.abs(A) ** 2
            
            def diff_operator_main(omega):
                return 1j * self.beta_2 / 2 * omega ** 2 - self.alpha_int / self.cavity_length / 2

            def body_func(carry, ind):
                key, A = carry
                key, subkey = jax.random.split(key, 2)
                
                A_1 = split_step_single_jax(A, self.time, self.cavity_length / 2, diff_operator_main, nl_operator_main, n_integral_iterations=n_per_step)

                ring_drop_phase = jnp.array([[1,0], [0, jnp.exp(-1j * detuning / 2)]])
                A_drop_next, A_2 = ring_drop_bs @ ring_drop_phase @ jnp.array([ _a_drop_in, A_1 ])
                
                A_3 = split_step_single_jax(A_2, self.time, self.cavity_length / 2, diff_operator_main, nl_operator_main, n_integral_iterations=n_per_step)
            
                ring_bus_phase = jnp.array([[1,0], [0, jnp.exp(-1j * detuning / 2)]])
            
                A_out_next, A_next = ring_bus_bs @ ring_bus_phase @ jnp.array([ _a_in,  A_3 ])
            
                noise_main = self.vacuum_noise * jnp.exp(2j * jnp.pi * jax.random.uniform(subkey, _a_init_main.shape))
                
                A_main_intra = A_next + noise_main
                
                return (key, A_main_intra), (A_main_intra, A_out_next, A_drop_next)
            
            carry, A_rest = jax.lax.scan(body_func, (_key_init, _a_init_main), jnp.arange(nrt))
            A_main_rest, A_out_rest, A_drop_rest = A_rest
            A_main = jnp.concat([_a_init_main[None,...], A_main_rest], axis=0)

            A_out = jnp.concat([_a_in[None,...], A_out_rest], axis=0)
            A_drop = jnp.concat([_a_drop_in[None,...], A_drop_rest], axis=0)
            return A_main, A_out, A_drop

        run_sim_jit = jax.jit(run_sim, static_argnames=('nrt','n_per_step'))
        return run_sim_jit

    def LLE_operator(self, A, A_in, detuning):
        '''
        This returns the RHS of the LLE 
        (so it is equvalent to `roundtrip time * dE/dT` where `T` is slow time)

        This method is mostly meant to be used for the steady state solver,
        but it is also useful for evaluating the steady state solver
        '''
        dispersion = jnp.fft.ifft( 
              1j * self.cavity_length * self.beta_2 / 2 * self.sim_omega ** 2 * jnp.fft.fft(A) 
          )
        nonlinearity = 1j * self.gamma * self.cavity_length * jnp.abs(A) ** 2 * A
        linear = ( - self.alpha - 1j * detuning ) * A + jnp.sqrt(self.coupling_ring_bus) * A_in
        return linear + nonlinearity + dispersion

    def get_steady_state_solver(self):
        '''
        Returns a jax compiled steady state solver. Uitilizes Newton-Raphson to
        find field steady state
        
        Note that this function needs to be re-ran for a change in the class 
        variables to be registered
        '''
        def run_solver(A_init, A_in, detuning, n_iter):
            '''
            Returns:
                Field in main ring
                RMS of the residuals of each field
            '''
            LLE = jax.tree_util.Partial(self.LLE_operator, A_in=A_in, detuning=detuning)

            def body_func(carry, ind):
                A = carry
                LLE_residual, LLE_jacobian = jax.vjp(LLE, A)
                I = jnp.eye(self.time.size, dtype=complex)
                J = jax.vmap(LLE_jacobian)(I)[0]

                delta_A = jnp.linalg.solve(-J, LLE_residual)
                error = jnp.sqrt(jnp.mean(jnp.abs(delta_A) ** 2))
                A = delta_A + A
                return A, (A, error)

            last, rest= jax.lax.scan(body_func, A_init, jnp.arange(n_iter))
            A_rest, error = rest
            #return jnp.concat([A_init[None,...], A_rest], axis=0), error
            return A_rest, error

        steady_state_solver = jax.jit(run_solver, static_argnames=('n_iter'))
        return steady_state_solver


    '''
    TODO: update this to single ring
    def resonance_transmission_nonlinear(self, detuning, detuning_main, detuning_aux, A_main, A_aux):
        _alpha_nl_main = self.gamma * self.cavity_length_main * \
                jnp.fft.fft(jnp.abs(A_main) ** 2 * A_main)[0] / jnp.fft.fft(A_main)[0]
        _alpha_nl_aux = self.gamma * self.cavity_length_aux * \
                jnp.fft.fft(jnp.abs(A_aux) ** 2 * A_aux)[0] / jnp.fft.fft(A_aux)[0]

        _alpha_main = ( self.alpha_int_main * self.cavity_length_main + self.coupling_ring_bus ) / 2
        _alpha_aux = ( self.alpha_int_aux * self.cavity_length_aux + self.coupling_ring_drop ) / 2
        t = 1 - self.coupling_ring_bus / ( _alpha_main + 1j * (-detuning + (detuning_main - _alpha_nl_main)) + \
                self.coupling_ring_ring / (_alpha_aux + 1j * (-detuning + (detuning_aux - _alpha_nl_aux))) )
        return jnp.abs(t) ** 2

    def resonance_transmission_linear(self, detuning, detuning_main, detuning_aux):
        _alpha_main = ( self.alpha_int_main * self.cavity_length_main + self.coupling_ring_bus ) / 2
        _alpha_aux = ( self.alpha_int_aux * self.cavity_length_aux + self.coupling_ring_drop ) / 2
        t = 1 - self.coupling_ring_bus / ( _alpha_main + 1j * (detuning_main - detuning) + \
                self.coupling_ring_ring / (_alpha_aux + 1j * (detuning_aux - detuning)) )
        return jnp.abs(t) ** 2
    '''
