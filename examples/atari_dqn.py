
import gym
import numpy as np
import atari_wrappers as aw
from tensorflow.python.keras import layers, initializers, models,
from drlbox.trainer import make_trainer


'''
Make a properly wrapped Atari env
'''
def make_env(name, num_frames=4, act_steps=2):
    env = gym.make(name)
    env = aw.Preprocessor(env, shape=(84, 84))
    env = aw.HistoryStacker(env, num_frames, act_steps)
    env = aw.RewardClipper(env, -1.0, 1.0)
    env = aw.EpisodicLife(env)
    return env


'''
When a state is represented by a list of frames, this interface converts it
to a correctly shaped, correctly typed numpy array which can be fed into
the convolutional neural network.
'''
def state_to_input(state):
    return np.stack(state, axis=-1).astype(np.float32)


'''
Build a convolutional actor-critic net that is similar to the Nature paper one.
Input arguments:
    env:        Atari environment.
'''
def make_model(env):
    num_frames = len(env.observation_space.spaces)
    height, width = env.observation_space.spaces[0].shape
    input_shape = height, width, num_frames
    model = models.Sequential()

    model.layers.Input(shape=input_shape)
    model.add(layers.Conv2D(32, 8, 8, subsample=(4, 4)))
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(64, 4, 4, subsample=(2, 2)))
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(64, 3, 3, subsample=(1, 1)))
    model.add(layers.Activation('relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(512))
    model.add(layers.Activation('relu'))
    model.add(layers.Dense(env.action_space.n))
    model.add(layers.Activation('linear'))
    print(model.summary())

    return model


'''
DQN on Breakout-v0
'''
if __name__ == '__main__':
    trainer = make_trainer('dqn',
        env_maker=lambda: make_env('Pong-v0'),
        model_maker=make_model,
        state_to_input=state_to_input,
        num_parallel=1,
        train_steps=1000000,
        rollout_maxlen=4,
        batch_size=8,
        verbose=True,
        )
    trainer.run()

