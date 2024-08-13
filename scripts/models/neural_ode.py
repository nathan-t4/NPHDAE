import flax.linen as nn
import diffrax
import jax.numpy as jnp

class NeuralODE(nn.Module):
    ''' 
        Simple Neural ODE
         - https://github.com/patrick-kidger/diffrax/issues/115
         - https://github.com/google/flax/discussions/2891
    '''
    derivative_net: MLP
    solver: diffrax._solver = diffrax.Tsit5()

    @nn.compact
    def __call__(self, inputs, ts=[0,1]):
        if self.is_initializing():
            self.derivative_net(jnp.concatenate((inputs, jnp.array([0]))))
        derivative_net_params = self.derivative_net.variables

        def derivative_fn(t, y, params):
            input = jnp.concatenate((y, jnp.array(t).reshape(-1)))
            return self.derivative_net.apply(params, input)
        
        term = diffrax.ODETerm(derivative_fn)

        solution = diffrax.diffeqsolve(
            term,
            self.solver, 
            t0=ts[0],
            t1=ts[-1],
            dt0=ts[1]-ts[0],
            y0=inputs,
            args=derivative_net_params,
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
            saveat=diffrax.SaveAt(ts=ts))
                
        return solution.ys