
from tensorflow.contrib.kfac import optimizer as opt


class KfacOptimizerTV(opt.KfacOptimizer):

    def apply_gradients(self, grads_and_vars, *args, **kwargs):
        """Applies gradients to variables.
        Args:
          grads_and_vars: List of (gradient, variable) pairs.
          *args: Additional arguments for super.apply_gradients.
          **kwargs: Additional keyword arguments for super.apply_gradients.
        Returns:
          An `Operation` that applies the specified gradients.
        """
        grads, train_vars = zip(*grads_and_vars)
        grads_and_vars = list(zip(grads, self.variables))

        # Compute step.
        steps_and_vars = self._compute_update_steps(grads_and_vars)

        # Modify variables in train_vars instead of grads_and_vars
        zip_st = zip(steps_and_vars, train_vars)
        steps_and_vars = [(step, t_var) for (step, _), t_var in zip_st]

        # Update trainable variables with this step.
        # Note: this super is getting the grandparent class of this class.
        return super(opt.KfacOptimizer, self).apply_gradients(steps_and_vars,
                                                              *args, **kwargs)


