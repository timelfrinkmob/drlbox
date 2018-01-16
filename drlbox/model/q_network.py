
import gym.spaces
from tensorflow import keras
from drlbox.layers.noisy_dense import NoisyDenseIG


'''
Input arguments:
    state:          Model input;
    feature:        Output of the feature function;
    action_space:   Action space of the environment; Discrete;
'''
def q_network_model(state, feature, action_space, noisy=False):
    if not isinstance(action_space, gym.spaces.discrete.Discrete):
        raise ValueError('action_space must be discrete in Q network')
    if noisy:
        dense = NoisyDenseIG
    else:
        dense = keras.layers.Dense
    q_value = dense(action_space.n)(feature)
    return keras.models.Model(inputs=state, outputs=q_value)

