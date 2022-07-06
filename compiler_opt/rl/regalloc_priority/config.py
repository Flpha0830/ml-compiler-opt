import gin
import tensorflow as tf
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step
from compiler_opt.rl import feature_ops


@gin.configurable()
def get_regalloc_signature_spec():
    observation_spec = dict(
        (key, tf.TensorSpec(dtype=tf.int64, shape=2, name=key))
        for key in ('size', 'stage', 'mask'))
    observation_spec['weight'] = tf.TensorSpec(dtype=tf.float32, shape=2, name='weight')

    reward_spec = tf.TensorSpec(dtype=tf.float32, shape=(), name='reward')
    time_step_spec = time_step.time_step_spec(observation_spec, reward_spec)

    action_spec = tensor_spec.BoundedTensorSpec(
        dtype=tf.int64,
        shape=(),
        name='priority',
        minimum=0,
        maximum=1)

    return time_step_spec, action_spec


@gin.configurable
def get_observation_processing_layer_creator():
    def observation_processing_layer(obs_spec):
        """Creates the layer to process observation given obs_spec."""
        if obs_spec.name == 'mask':
            return tf.keras.layers.Lambda(feature_ops.discard_fn)

        if obs_spec.name in ('size', 'stage', 'weight'):
            return tf.keras.layers.Lambda(feature_ops.identity_fn)

        # Make sure all features have a preprocessing function.
        raise KeyError('Missing preprocessing function for some feature.')

    return observation_processing_layer
