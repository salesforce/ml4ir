"""
Implementation of the Lion optimizer.
Source : https://github.com/google/automl/blob/master/lion/lion_tf2.py
Paper : https://arxiv.org/pdf/2302.06675.pdf
"""

import tensorflow as tf


class Lion(tf.keras.optimizers.legacy.Optimizer):
    """Optimizer that implements the Lion algorithm."""

    def __init__(self,
                 learning_rate=0.0001,
                 beta_1=0.9,
                 beta_2=0.99,
                 wd=0,
                 name='lion',
                 **kwargs):
        """Construct a new Lion optimizer."""

        super(Lion, self).__init__(name, **kwargs)
        self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
        self._set_hyper('beta_1', beta_1)
        self._set_hyper('beta_2', beta_2)
        self._set_hyper('wd', wd)

    def _create_slots(self, var_list):
        # Create slots for the first and second moments.
        # Separate for-loops to respect the ordering of slot variables from v1.
        for var in var_list:
            self.add_slot(var, 'm')

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(Lion, self)._prepare_local(var_device, var_dtype, apply_state)

        beta_1_t = tf.identity(self._get_hyper('beta_1', var_dtype))
        beta_2_t = tf.identity(self._get_hyper('beta_2', var_dtype))
        wd_t = tf.identity(self._get_hyper('wd', var_dtype))
        lr = apply_state[(var_device, var_dtype)]['lr_t']
        apply_state[(var_device, var_dtype)].update(
            dict(
                lr=lr,
                beta_1_t=beta_1_t,
                one_minus_beta_1_t=1 - beta_1_t,
                beta_2_t=beta_2_t,
                one_minus_beta_2_t=1 - beta_2_t,
                wd_t=wd_t))

    @tf.function(jit_compile=True)
    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype)) or
                        self._fallback_apply_state(var_device, var_dtype))

        m = self.get_slot(var, 'm')
        var_t = var.assign_sub(
            coefficients['lr_t'] *
            (tf.math.sign(m * coefficients['beta_1_t'] +
                          grad * coefficients['one_minus_beta_1_t']) +
             var * coefficients['wd_t']))
        with tf.control_dependencies([var_t]):
            m.assign(m * coefficients['beta_2_t'] +
                     grad * coefficients['one_minus_beta_2_t'])

    @tf.function(jit_compile=True)
    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype)) or
                        self._fallback_apply_state(var_device, var_dtype))

        m = self.get_slot(var, 'm')
        m_t = m.assign(m * coefficients['beta_1_t'])
        m_scaled_g_values = grad * coefficients['one_minus_beta_1_t']
        m_t = m_t.scatter_add(tf.IndexedSlices(m_scaled_g_values, indices))
        var_t = var.assign_sub(coefficients['lr'] *
                               (tf.math.sign(m_t) + var * coefficients['wd_t']))

        with tf.control_dependencies([var_t]):
            m_t = m_t.scatter_add(tf.IndexedSlices(-m_scaled_g_values, indices))
            m_t = m_t.assign(m_t * coefficients['beta_2_t'] /
                             coefficients['beta_1_t'])
            m_scaled_g_values = grad * coefficients['one_minus_beta_2_t']
            m_t.scatter_add(tf.IndexedSlices(m_scaled_g_values, indices))

    def get_config(self):
        config = super(Lion, self).get_config()
        config.update({
            'learning_rate': self._serialize_hyperparameter('learning_rate'),
            'beta_1': self._serialize_hyperparameter('beta_1'),
            'beta_2': self._serialize_hyperparameter('beta_2'),
            'wd': self._serialize_hyperparameter('wd'),
        })
        return config
