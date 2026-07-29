from .split_step_1d_jax import split_step_single_jax
import jax
import jax.numpy as jnp

c = 299792458
hbar = 1.054571817e-34 # J s

class CoupledRings:
    def __init__(self, *, neff, FSR_main, FSR_aux, coupling_ring_bus, 
             coupling_ring_ring, coupling_ring_drop, gamma, alpha_int_main, 
             alpha_int_aux, beta_2, omega_0, vacuum_noise=None, n_small_time=512):
        self.neff = neff,
        self.FSR_main = FSR_main
        self.FSR_aux = FSR_aux
        self.coupling_ring_bus = coupling_ring_bus
        self.coupling_ring_ring = coupling_ring_ring
        self.coupling_ring_drop = coupling_ring_drop
        self.gamma = gamma
        self.alpha_int_main = alpha_int_main
        self.alpha_int_aux = alpha_int_aux
        self.beta_2 = beta_2
        self.omega_0 = omega_0

        if vacuum_noise is None:
            self.vacuum_noise = jnp.sqrt(hbar * omega_0 * FSR_main)

        _t_rt = 1 / FSR_main
        self.time = jnp.linspace(-_t_rt / 2, _t_rt / 2, n_small_time)

        self.frequency = jnp.fft.fftfreq(n_small_time, self.time[1] - self.time[0]) + omega_0 / ( 2 * jnp.pi )
        self.wavelength = c / self.frequency

        self.cavity_length_main = c / ( neff * FSR_main )
        self.cavity_length_aux = c / ( neff * FSR_aux )
        self.cavity_ratio = self.cavity_length_aux / self.cavity_length_main
        self.drift = (self.cavity_length_aux / (c / neff) - self.cavity_length_main / (c / neff)) / self.cavity_length_main

    def get_ikeda_map(self):
        ring_bus_bs = jnp.array([[jnp.sqrt(1 - self.coupling_ring_bus), 1j * jnp.sqrt(self.coupling_ring_bus)], 
                                [1j * jnp.sqrt(self.coupling_ring_bus), jnp.sqrt(1 - self.coupling_ring_bus)]])

        ring_ring_bs = jnp.array([[jnp.sqrt(1 - self.coupling_ring_ring), 1j * jnp.sqrt(self.coupling_ring_ring)], 
                                 [1j * jnp.sqrt(self.coupling_ring_ring), jnp.sqrt(1 - self.coupling_ring_ring)]])

        ring_drop_bs = jnp.array([[jnp.sqrt(1 - self.coupling_ring_drop), 1j * jnp.sqrt(self.coupling_ring_drop)], 
                                [1j * jnp.sqrt(self.coupling_ring_drop), jnp.sqrt(1 - self.coupling_ring_drop)]])

        def run_sim(A_init_main, A_init_aux, Ain, detuning_main, detuning_aux, nrt, n_per_step=2):
            '''
            Returns:
                Field in main ring
                Field in aux ring
                Field coupled out into the bus waveguide
                Field coupled out into the drop waveguide
            '''
            _key_init = jax.random.key(0)
            _a_init_main = jnp.array(A_init_main, dtype=jnp.complex64)
            _a_init_aux = jnp.array(A_init_aux, dtype=jnp.complex64)

            _a_in = jnp.array(Ain, dtype=jnp.complex64)
            _a_drop_in = jnp.zeros(Ain.shape, dtype=jnp.complex64)

            def nl_operator_main(A):
                return 1j * self.gamma * jnp.abs(A) ** 2
            
            def nl_operator_aux(A):
                return 1j * self.cavity_ratio * self.gamma * jnp.abs(A) ** 2
            
            def diff_operator_main(omega):
                return 1j * self.beta_2 / 2 * omega ** 2 - self.alpha_int_main / 2
            
            def diff_operator_aux(omega):
                return 1j * self.cavity_ratio * self.beta_2 / 2 * omega ** 2 - self.cavity_ratio * self.alpha_int_aux / 2 + 1j * omega * self.drift

            def body_func(carry, ind):
                key, A_main, A_aux = carry
                key, subkey1, subkey2 = jax.random.split(key, 3)
                
                A_main_1 = split_step_single_jax(A_main, self.time, self.cavity_length_main / 2, diff_operator_main, nl_operator_main, n_integral_iterations=n_per_step)
                A_aux_1  = split_step_single_jax(A_aux,  self.time, self.cavity_length_main / 2, diff_operator_aux, nl_operator_aux, n_integral_iterations=n_per_step)

                ring_ring_phase = jnp.array([[jnp.exp(-1j * detuning_aux / 2), 0],
                                    [0, jnp.exp(-1j * detuning_main / 2)]])
            
                A_aux_2, A_main_2 = ring_ring_bs @ ring_ring_phase @ jnp.array([ A_aux_1, A_main_1 ])
                
                A_main_3 =  split_step_single_jax(A_main_2, self.time, self.cavity_length_main / 2, diff_operator_main, nl_operator_main, n_integral_iterations=n_per_step)
                A_aux_3  =  split_step_single_jax(A_aux_2,  self.time, self.cavity_length_main / 2, diff_operator_aux,  nl_operator_aux,  n_integral_iterations=n_per_step)
                
                ring_drop_phase = jnp.array([[1,0], [0, jnp.exp(-1j * detuning_aux / 2)]])
            
                A_drop_next, A_aux_next = ring_drop_bs @ ring_drop_phase @ jnp.array([ _a_drop_in, A_aux_3 ])
            
                ring_bus_phase = jnp.array([[1,0], [0, jnp.exp(-1j * detuning_main / 2)]])
            
                A_out_next, A_main_next = ring_bus_bs @ ring_bus_phase @ jnp.array([ _a_in,  A_main_3 ])
            
                noise_main = self.vacuum_noise * jnp.exp(2j * jnp.pi * jax.random.uniform(subkey1, _a_init_main.shape))
                noise_aux =  self.vacuum_noise * jnp.exp(2j * jnp.pi * jax.random.uniform(subkey2, _a_init_aux.shape))
                
                A_main_intra = A_main_next + noise_main
                A_aux_intra = A_aux_next + noise_aux
                
                return (key, A_main_intra, A_aux_intra), (A_main_intra, A_aux_intra, A_out_next, A_drop_next)
            
            carry, A_rest = jax.lax.scan(body_func, (_key_init, _a_init_main, _a_init_aux), jnp.arange(nrt))
            A_main_rest, A_aux_rest, A_out_rest, A_drop_rest = A_rest
            A_main = jnp.concat([_a_init_main[None,...], A_main_rest], axis=0)
            A_aux = jnp.concat([_a_init_aux[None,...], A_aux_rest], axis=0)

            A_out = jnp.concat([_a_in[None,...], A_out_rest], axis=0)
            A_drop = jnp.concat([_a_drop_in[None,...], A_drop_rest], axis=0)
            return A_main, A_aux, A_out, A_drop

        run_sim_jit = jax.jit(run_sim, static_argnames=('nrt','n_per_step'))
        return run_sim_jit

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


def make_coupled_ring_simulation(*, neff, FSR_main, FSR_aux, coupling_ring_bus, 
             coupling_ring_ring, coupling_ring_drop, gamma, alpha_int_main, 
             alpha_int_aux, beta_2, omega_0=None, vacuum_noise=None, n_small_time=512):
    '''
    '''
    if omega_0 is None and vacuum_noise is None:
        raise ValueError('omega_0 is used to calculate vacuum noise. You cannot exclude both')

    if vacuum_noise is None:
        vacuum_noise = jnp.sqrt(hbar * omega_0 * FSR_main)

    _t_rt = 1 / FSR_main
    time = jnp.linspace(-_t_rt / 2, _t_rt / 2, n_small_time)

    cavity_length_main = c / ( neff * FSR_main )
    cavity_length_aux = c / ( neff * FSR_aux )
    cavity_ratio = cavity_length_aux / cavity_length_main
    drift = (cavity_length_aux / (c / neff) - cavity_length_main / (c / neff)) / cavity_length_main

    ring_bus_bs = jnp.array([[jnp.sqrt(1 - coupling_ring_bus), 1j * jnp.sqrt(coupling_ring_bus)], 
                            [1j * jnp.sqrt(coupling_ring_bus), jnp.sqrt(1 - coupling_ring_bus)]])

    ring_ring_bs = jnp.array([[jnp.sqrt(1 - coupling_ring_ring), 1j * jnp.sqrt(coupling_ring_ring)], 
                             [1j * jnp.sqrt(coupling_ring_ring), jnp.sqrt(1 - coupling_ring_ring)]])

    ring_drop_bs = jnp.array([[jnp.sqrt(1 - coupling_ring_drop), 1j * jnp.sqrt(coupling_ring_drop)], 
                            [1j * jnp.sqrt(coupling_ring_drop), jnp.sqrt(1 - coupling_ring_drop)]])

    def run_sim(A_init_main, A_init_aux, Ain, nrt, detuning_main, detuning_aux):
        _key_init = jax.random.key(0)
        _a_init_main = jnp.array(A_init_main, dtype=jnp.complex64)
        _a_init_aux = jnp.array(A_init_aux, dtype=jnp.complex64)

        _a_in = jnp.array(Ain, dtype=jnp.complex64)
        _a_drop_in = jnp.zeros(Ain.shape, dtype=jnp.complex64)

        def nl_operator_main(A):
            return 1j * gamma * jnp.abs(A) ** 2
        
        def nl_operator_aux(A):
            return 1j * cavity_ratio * gamma * jnp.abs(A) ** 2
        
        def diff_operator_main(omega):
            return 1j * beta_2 / 2 * omega ** 2 - alpha_int_main / 2
        
        def diff_operator_aux(omega):
            return 1j * cavity_ratio * beta_2 / 2 * omega ** 2 - cavity_ratio * alpha_int_aux / 2 + 1j * omega * drift

        def body_func(carry, ind):
            key, A_main, A_aux = carry
            key, subkey1, subkey2 = jax.random.split(key, 3)
            
            A_main_1 = split_step_single_jax(A_main, time, cavity_length_main / 2, diff_operator_main, nl_operator_main, n_integral_iterations=2)
            A_aux_1  = split_step_single_jax(A_aux,  time, cavity_length_main / 2, diff_operator_aux, nl_operator_aux, n_integral_iterations=2)

            ring_ring_phase = jnp.array([[jnp.exp(-1j * detuning_aux / 2), 0],
                                [0, jnp.exp(-1j * detuning_main / 2)]])
        
            A_aux_2, A_main_2 = ring_ring_bs @ ring_ring_phase @ jnp.array([ A_aux_1, A_main_1 ])
            
            A_main_3 =  split_step_single_jax(A_main_2, time, cavity_length_main / 2, diff_operator_main, nl_operator_main, n_integral_iterations=2)
            A_aux_3  =  split_step_single_jax(A_aux_2,  time, cavity_length_main / 2, diff_operator_aux,  nl_operator_aux,  n_integral_iterations=2)
            
            ring_drop_phase = jnp.array([[1,0], [0, jnp.exp(-1j * detuning_aux / 2)]])
        
            A_drop_next, A_aux_next = ring_drop_bs @ ring_drop_phase @ jnp.array([ _a_drop_in, A_aux_3 ])
        
            ring_bus_phase = jnp.array([[1,0], [0, jnp.exp(-1j * detuning_main / 2)]])
        
            A_out_next, A_main_next = ring_bus_bs @ ring_bus_phase @ jnp.array([ _a_in,  A_main_3 ])
        
            noise_main = vacuum_noise * jnp.exp(2j * jnp.pi * jax.random.uniform(subkey1, _a_init_main.shape))
            noise_aux = vacuum_noise * jnp.exp(2j * jnp.pi * jax.random.uniform(subkey2, _a_init_aux.shape))
            
            A_main_intra = A_main_next + noise_main
            A_aux_intra = A_aux_next + noise_aux
            
            return (key, A_main_intra, A_aux_intra), (A_main_intra, A_aux_intra)
        
        carry, A_rest = jax.lax.scan(body_func, (_key_init, _a_init_main, _a_init_aux), jnp.arange(nrt))
        A_main_rest, A_aux_rest = A_rest[0], A_rest[1]
        A_main = jnp.concat([_a_init_main[None,...], A_main_rest], axis=0)
        A_aux = jnp.concat([_a_init_aux[None,...], A_aux_rest], axis=0)
        return A_main, A_aux

    run_sim_jit = jax.jit(run_sim, static_argnames=('nrt',))
    return run_sim_jit
