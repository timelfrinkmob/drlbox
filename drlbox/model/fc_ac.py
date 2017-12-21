
import gym.spaces
from tensorflow.contrib.keras import layers, models, initializers


def state_to_input(state):
    return state.ravel()

'''
Input arguments:
    observation_space: Observation space of the environment;
    action_space:      Action space of the environment;
    arch_str:          Architecture of the actor-critic net, e.g., '16 16 16'.
'''
def model(observation_space, action_space, arch_str):
    net_arch = arch_str.split(' ')
    net_arch = [int(num) for num in net_arch]
    state = layers.Input(shape=observation_space.shape)
    feature = state
    for num_hid in net_arch:
        feature = layers.Dense(num_hid, activation='relu')(feature)
    if isinstance(action_space, gym.spaces.discrete.Discrete): # discrete action
        size_logits = action_space.n
        action_mode = 'discrete'
    elif isinstance(action_space, gym.spaces.box.Box): # continuous action
        size_logits = len(action_space.shape) + 1
        action_mode = 'continuous'
    else:
        raise ValueError('type of action_space is illegal')
    near_zeros = initializers.RandomNormal(stddev=1e-3)
    logits = layers.Dense(size_logits, kernel_initializer=near_zeros)(feature)
    value = layers.Dense(1)(feature)
    model = models.Model(inputs=state, outputs=[value, logits])
    model.action_mode = action_mode
    return model

