import abc
import jax
import diffrax
import jax.numpy as jnp
from integrators.euler_variants import euler, semi_implicit_euler
from integrators.rk4 import rk4

class System(abc.ABC):
    def __init__(self, dt, num_mp_steps, norm_stats, integration_method):
        self.dt = dt
        self.num_mp_steps = num_mp_steps
        self.norm_stats = norm_stats
        self.integration_method = integration_method

    @abc.abstractmethod
    def dynamics_function(self, state, t, graph):
        """ return next state """

    @abc.abstractmethod
    def eval_plots(T, pred_data, exp_data):
        "Save evaluation plots"
    
class MassSpringIntegrator(System):
    def __init__(self, dt, num_mp_steps, norm_stats, integration_method='SemiImplicitEuler'):
        super().__init__(dt, num_mp_steps, norm_stats, integration_method)
    
    def dynamics_function(self, state, t, graph):
        cur_pos = jnp.array([state[0]])
        cur_vel = jnp.array([state[1]])

        def force(t, args):
            del t, args
            normalized_acc = graph.nodes
            pred_acc = normalized_acc * self.norm_stats.acceleration.std + self.norm_stats.acceleration.mean
            return pred_acc
        
        def newtons_equation_of_motion(t, y, args):
            """
                TODO: generalize to n-dim systems 
                y = [x0, v0, x1, v1] 

                A = np.zeros((2*N, 2*N)) 
                ...
            """
            A = jnp.array([[0, 1, 0, 0], 
                           [0, 0, 0, 0], 
                           [0, 0, 0, 1], 
                           [0, 0, 0, 0]])
            F_ext = force(t, args)
            # TODO: check if there is bug here!
            F = jnp.concatenate((jnp.array([0]), F_ext[0], jnp.zeros(1), F_ext[1])) 
            return A @ y + F
        
        @jax.jit
        def solve(y0, args):
            t0 = 0
            t1 = self.num_mp_steps * self.dt
            dt0 = self.dt
            if self.integration_method == 'Euler':
                term = diffrax.ODETerm(newtons_equation_of_motion)
                solver = diffrax.Euler()
                sol = diffrax.diffeqsolve(term, solver, t0, t1, dt0, y0, args=args)
                next_pos = sol.ys[-1, 0:2]
                next_vel = sol.ys[-1, 2:4]
            elif self.integration_method == 'Tsit5':
                term = diffrax.ODETerm(newtons_equation_of_motion)
                solver = diffrax.Tsit5()
                sol = diffrax.diffeqsolve(term, solver, t0, t1, dt0, y0, args=args)
                next_vel = sol.ys[-1, 2:4]
            elif self.integration_method == 'SemiImplicitEuler':
                pred_acc = force(t0, args).squeeze()
                next_vel = cur_vel + pred_acc * (dt0 * self.num_mp_steps)
                next_pos = cur_pos + next_vel * (dt0 * self.num_mp_steps)
            else:
                raise NotImplementedError('Invalid integration method')    
            return next_pos, next_vel
        
        y0 = jnp.concatenate((cur_pos, cur_vel), axis=0).reshape(-1, 1)
        args = None
        next_pos, next_vel = solve(y0, args)  
        pred_acc = force(0, args).squeeze()

        return next_pos, next_vel, pred_acc

class LCIntegrator(System):
    def __init__(self, dt, num_mp_steps, norm_stats, integration_method='rk4'):
        super().__init__(dt, num_mp_steps, norm_stats, integration_method)

    def dynamics_function(self, state, H_grad, t):
        def port_hamiltonian_dynamics(state, t):
            J = jnp.array([[0, 1],
                           [-1, 0]])
            return jnp.matmul(J, H_grad)
        
        # integrate dynamics
        if self.integration_method == 'rk4':
            next_state = rk4(port_hamiltonian_dynamics, state, t, self.dt)
        elif self.integration_method == 'euler':
            next_state = euler(port_hamiltonian_dynamics, state, t, self.dt)
        else:
            raise NotImplementedError('Unsupported integration method for LC circuit dynamics')
        
        return next_state
    
class CLCIntegrator(System):
    def __init__(self, dt, num_mp_steps, norm_stats, integration_method='SemiImplicitEuler'):
        super().__init__(dt, num_mp_steps, norm_stats, integration_method)
    
    def dynamics_function(self, state, graph):
        return super().dynamics_function(state)
    
class CoupledLCIntegrator(System):
    def __init__(self, dt, num_mp_steps, norm_stats, integration_method='SemiImplicitEuler'):
        super().__init__(dt, num_mp_steps, norm_stats, integration_method)
    
    def dynamics_function(self, state, graph):
        return super().dynamics_function(state)