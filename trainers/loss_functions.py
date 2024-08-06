import jax
import jax.numpy as jnp

def l2_loss_constructor(model, loss_function_setup):

    forward = model.forward
    pen_l2_nn_params = float(loss_function_setup['pen_l2_nn_params'])

    @jax.jit
    def loss(params, 
            x : jnp.ndarray, 
            y : jnp.ndarray) -> jnp.float32:

        out = forward(params, x)
        num_datapoints = x.shape[0]
        
        data_loss = jnp.sum((out - y)**2) / num_datapoints
        normalized_data_loss = data_loss / (jnp.sum(y**2) / num_datapoints)

        regularization_loss = pen_l2_nn_params * \
            sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params))

        total_loss = data_loss + regularization_loss

        # Build a dictionary of the breakdown of the loss function.
        loss_vals = {
            'total_loss' : total_loss,
            'data_loss' : data_loss,
            'regularization_loss' : regularization_loss,
            'normalized_data_loss' : normalized_data_loss
        }

        return total_loss, loss_vals

    return loss

def l2_and_g_loss_constructor(model, loss_function_setup):

    forward = model.forward
    forward_g = model.forward_g

    pen_l2 = float(loss_function_setup['pen_l2'])
    pen_g = float(loss_function_setup['pen_g'])
    pen_l2_nn_params = float(loss_function_setup['pen_l2_nn_params'])

    @jax.jit
    def loss(params, 
            x : jnp.ndarray, 
            y : jnp.ndarray) -> jnp.float32:

        out = forward(params, x)
        out_g = forward_g(params, x) # TODO: Check if this is wrong.
        num_datapoints = x.shape[0]
        
        data_loss = pen_l2 * jnp.sum((out - y)**2) / num_datapoints
        normalized_data_loss = data_loss / (jnp.sum(y**2) / num_datapoints)

        regularization_loss = pen_l2_nn_params * \
            sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params))
        
        g_loss = pen_g * jnp.sum((out_g)**2) / num_datapoints

        total_loss = data_loss + g_loss + regularization_loss

        # Build a dictionary of the breakdown of the loss function.
        loss_vals = {
            'total_loss' : total_loss,
            'data_loss' : data_loss,
            'g_loss' : g_loss,
            'regularization_loss' : regularization_loss,
            'normalized_data_loss' : normalized_data_loss
        }

        return total_loss, loss_vals

    return loss

###############################################################################
loss_function_factory ={
    'l2_loss' : l2_loss_constructor,
    'l2_and_g_loss' : l2_and_g_loss_constructor,
}

def get_loss_function(model, loss_function_setup):
    return loss_function_factory[loss_function_setup['loss_function_type']](model, loss_function_setup)