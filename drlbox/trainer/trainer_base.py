
from multiprocessing import Process, Event, cpu_count
import socket

import os
import signal
import time
from datetime import timedelta

import tensorflow as tf
import numpy as np
from drlbox.layer.noisy_dense import NoisyDenseIG, NoisyDenseFG
from drlbox.net.kfac.optimizer import KfacOptimizerTV
from drlbox.net.kfac.build_layer_collection import build_layer_collection
from drlbox.common.replay import Replay, PriorityReplay
from drlbox.common.util import discrete_action, continuous_action
from drlbox.common.tasker import Tasker
from .step_counter import StepCounter
from .rollout import Rollout


'''
Explanation of batched n-step training and arguments:

1. Rollout:
    The basic unit of training is a length-L "rollout" of the form
    { s_t, a_t, r_t, s_{t+1}, ..., s_{t+L} }
    which contains L transitions.  In practice, L is not always a fixed length
    as a rollout must terminate at the end of an episode.

    Argument 'rollout_maxlen' is the maximum length a rollout can ever have,
    and is related with the number of bootstrap steps.  For example, setting
    'rollout_maxlen = 1' corresponds to 1-step bootstrapped TD learning.
    If we set 'rollout_maxlen = N', then the first state in the rollout will be
    subject to a N-step TD learning, the second state will be subject to
    a (N-1)-step TD learning, and so on.

2. Rollout list:
    "rollout_list" (abbr. "rlist") is a list of (various-length) rollouts
    and is guaranteed to contain a fixed number of transitions.

    Argument 'rollout_maxlen' is also this fixed number.

3. Batch:
    A "batch" is simply a fixed number of rollout lists.  One training on
    a single batch executes exactly one update to the network weights.

    Argument 'batch_size' is the number of rollout lists.

'''

LOCALHOST = 'localhost'
JOBNAME = 'local'

'''
Optimizer related default kwargs
'''
ADAM_KWARGS = dict(
    learning_rate=1e-4,
    epsilon=1e-4,
    )
KFAC_KWARGS = dict(
    learning_rate=1e-4,
    cov_ema_decay=0.95,
    damping=1e-3,
    norm_constraint=1e-3,
    momentum=0.0,
    )

'''
Replay memory related default kwargs
'''
REPLAY_KWARGS = dict(
    maxlen=1000,
    minlen=100,
    )

'''
Trainer default kwargs
'''
TRAINER_KWARGS = dict(
    feature_maker=None,
    model_maker=None,           # if set, ignores feature_maker
    num_parallel=cpu_count(),
    port_begin=2220,
    discount=0.99,
    train_steps=1000000,
    rollout_maxlen=32,
    batch_size=1,
    online_learning=True,       # whether or not to perform online learning
    replay_type=None,           # None, 'uniform', or 'prioritized'
    replay_ratio=4,
    replay_kwargs={},
    optimizer='adam',           # 'adam', 'kfac', or tf.train.Optimizer instance
    opt_clip_norm=40.0,
    opt_kwargs={},
    kfac_inv_upd_interval=10,
    noisynet=None,              # None, 'ig', or 'fg'
    save_dir=None,              # directory to save tf.keras models
    save_interval=10000,
    catch_signal=False,         # effective on multiprocessing only
    )


class Trainer(Tasker):

    dense_layer = tf.keras.layers.Dense
    KWARGS = {**Tasker.KWARGS, **TRAINER_KWARGS}

    def run(self):
        # change default dense_layer to noisy layer if requested
        if self.noisynet is None:
            pass
        elif self.noisynet == 'ig':
            self.dense_layer = NoisyDenseIG
            self.print('Using independent Gaussian NoisyNet')
        elif self.noisynet == 'fg':
            self.dense_layer = NoisyDenseFG
            self.print('Using factorized Gaussian NoisyNet')
        else:
            raise ValueError('noisynet={} is invalid'.format(self.noisynet))

        self.port_list = [self.port_begin + i for i in range(self.num_parallel)]

        # single process
        if self.num_parallel == 1:
            self.worker(0)
            return

        # multiprocess parallel training
        for port in self.port_list:
            if not self.port_available(LOCALHOST, port):
                raise NameError('port {} is not available'.format(port))
        self.print('Claiming {} port {} ...'.format(LOCALHOST, self.port_list))
        self.event_finished = Event()
        self.worker_list = []
        try:
            for wid in range(self.num_parallel):
                worker = Process(target=self.worker, args=(wid,))
                worker.start()
                self.worker_list.append(worker)
        except:
            self.terminate_workers()

        # set handlers if requested
        if self.catch_signal:
            self.default_sigint_handler = signal.signal(signal.SIGINT,
                                                        self.signal_handler)
            self.default_sigterm_handler = signal.signal(signal.SIGTERM,
                                                         self.signal_handler)
            self.print('SIGINT and SIGTERM will be caught by drlbox')

        # terminates the entire training when the master worker terminates
        master_worker = self.worker_list[0]
        wait_counter = 0
        start_time = time.time()
        while not self.event_finished.is_set():
            wait_counter += 1
            if wait_counter >= 3000:
                wait_counter = 0
                elapsed = int(time.time() - start_time)
                time_str = str(timedelta(seconds=elapsed))
                self.print('Elapsed time:', time_str)
            time.sleep(0.1)
        self.print('A worker just terminated -- training should end soon')
        self.terminate_workers()

        # restore default handlers
        if self.catch_signal:
            signal.signal(signal.SIGINT, self.default_sigint_handler)
            signal.signal(signal.SIGTERM, self.default_sigterm_handler)
            self.print('SIGINT and SIGTERM default handlers reset to default')
        self.print('Asynchronous training has ended')

    def signal_handler(self, signum, frame):
        self.event_finished.set()

    def terminate_workers(self):
        for worker in self.worker_list[::-1]:
            while worker.is_alive():
                worker.terminate()
                time.sleep(0.01)

    def worker(self, wid):
        assert callable(self.env_maker)
        env = self.env_maker()

        # determine action mode from env.action_space
        if discrete_action(env.action_space):
            self.action_mode = 'discrete'
            self.action_dim = env.action_space.n
        elif continuous_action(env.action_space):
            self.action_mode = 'continuous'
            self.action_dim = len(env.action_space.shape)
            self.action_low = env.action_space.low
            self.action_high = env.action_space.high
        else:
            raise TypeError('Invalid type of env.action_space')

        self.is_master = wid == 0
        if self.is_master and self.save_dir is not None:
            env_name = 'UnknownEnv-v0' if env.spec is None else env.spec.id
            self.output = self.get_output_dir(env_name)
        else:
            self.output = None

        # ports, cluster, and server
        cluster_list = ['{}:{}'.format(LOCALHOST, p) for p in self.port_list]
        cluster = tf.train.ClusterSpec({JOBNAME: cluster_list})
        server = tf.train.Server(cluster, job_name=JOBNAME, task_index=wid)
        self.print('Starting server #{}'.format(wid))

        self.setup_algorithm()

        # global/local devices
        worker_dev = '/job:{}/task:{}/cpu:0'.format(JOBNAME, wid)
        rep_dev = tf.train.replica_device_setter(worker_device=worker_dev,
                                                 cluster=cluster)

        self.setup_nets(worker_dev, rep_dev, env)
        if self.replay_type is None:
            self.replay = None
        else:
            replay_kwargs = {**REPLAY_KWARGS, **self.replay_kwargs}
            if self.is_master:
                self.print_kwargs(replay_kwargs, 'Replay memory arguments')
            if self.replay_type == 'uniform':
                self.replay = Replay(**replay_kwargs)
            elif self.replay_type == 'prioritized':
                self.replay = PriorityReplay(**replay_kwargs)
            else:
                message = 'replay type {} invalid'.format(self.replay_type)
                raise ValueError(message)

        # begin tensorflow session, build async RL agent and train
        port = self.port_list[wid]
        with tf.Session('grpc://{}:{}'.format(LOCALHOST, port)) as sess:
            sess.run(tf.global_variables_initializer())
            self.set_session(sess)

            # train the agent
            self.train_on_env(env)

        if self.num_parallel > 1:
            self.event_finished.set()
            if self.is_master:
                while True:
                    time.sleep(1)

    def train_on_env(self, env):
        step = self.step_counter.step_count()
        if self.is_master:
            last_save = step
            self.save_model(step)

        state = env.reset()
        ep_reward = 0.0
        while step <= self.train_steps:
            self.sync_to_global()
            batch = []
            for batch_step in range(self.batch_size):
                rlist = [Rollout(state)]
                for rlist_step in range(self.rollout_maxlen):
                    net_input = self.state_to_input(state)
                    act_val = self.online_net.action_values([net_input])[0]
                    action = self.policy.select_action(act_val)
                    state, reward, done, info = env.step(action)
                    ep_reward += reward
                    rlist[-1].append(state, action, reward, done, act_val)
                    if done:
                        state = env.reset()
                        if rlist_step < self.rollout_maxlen - 1:
                            rlist.append(Rollout(state))
                        self.print('episode reward {:5.2f}'.format(ep_reward))
                        ep_reward = 0.0
                    if len(rlist[-1]) >= self.rollout_maxlen:
                        if rlist_step < self.rollout_maxlen - 1:
                            rlist.append(Rollout(state))
                batch.append(rlist)

            if self.online_learning:
                # on-policy training on the newly collected rollout list
                batch_loss, batch_error = self.train_on_batch(batch)
                batch_loss_list = [batch_loss]
            else:
                batch_loss_list = []

            # off-policy training if there is a memory
            if self.replay is not None:
                if self.replay_type == 'prioritized' and self.online_learning:
                    self.replay.extend(batch, batch_error)
                else:
                    self.replay.extend(batch)
                if self.replay.usable():
                    rep = np.random.poisson(self.replay_ratio)
                    for _ in range(rep):
                        batch, idx, weight = self.replay.sample(self.batch_size)
                        self.sync_to_global()
                        if self.replay_type == 'prioritized':
                            args = batch, weight
                        else:
                            args = batch,
                        batch_loss, batch_error = self.train_on_batch(*args)
                        batch_loss_list.append(batch_loss)
                        if self.replay_type == 'prioritized':
                            self.replay.update_priority(idx, batch_error)

            # step, print, etc.
            self.step_counter.increment(self.batch_size * self.rollout_maxlen)
            step = self.step_counter.step_count()
            if self.is_master:
                if step - last_save > self.save_interval:
                    self.save_model(step)
                    last_save = step
                str_step = 'training step {}/{}'.format(step, self.train_steps)
                mean_batch_loss = np.mean(batch_loss_list)
                self.print(str_step + ', loss {:3.3f}'.format(mean_batch_loss))
        # save at the end of training
        if self.is_master:
            self.save_model(step)

    def port_available(self, host, port):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        return sock.connect_ex((host, port)) != 0

    def build_net(self, env=None, is_global=False):
        net = self.net_cls()
        if self.load_model is not None:
            if is_global:
                model = self.do_load_model()
                self.saved_weights = model.get_weights()
            else:
                model = self.do_load_model(load_weights=False)
        else:
            if self.model_maker is None:
                assert callable(self.feature_maker)
                state, feature = self.feature_maker(env.observation_space)
                model = self.build_model(state, feature, **self.model_kwargs)
            else:
                assert callable(self.model_maker)
                model = self.model_maker(env)
        net.set_model(model)
        if self.noisynet is not None:
            net.set_noise_list()
        return net

    def layer_flatten(self, layer):
        if len(layer.shape) > 2:
            layer = tf.keras.layers.Flatten()(layer)
        return layer

    def set_online_optimizer(self):
        if self.optimizer == 'adam':
            adam_kwargs = {**ADAM_KWARGS, **self.opt_kwargs}
            if self.is_master:
                self.print_kwargs(adam_kwargs, 'Adam arguments')
            adam = tf.train.AdamOptimizer(**adam_kwargs)
            self.online_net.set_optimizer(adam, clip_norm=self.opt_clip_norm,
                                          train_weights=self.global_net.weights)
        elif self.optimizer == 'kfac':
            kfac_kwargs = {**KFAC_KWARGS, **self.opt_kwargs}
            if self.is_master:
                self.print_kwargs(kfac_kwargs, 'KFAC arguments')
            layer_collection = build_layer_collection(
                layer_list=self.online_net.model.layers,
                loss_list=self.online_net.kfac_loss_list,
                )
            kfac = KfacOptimizerTV(**kfac_kwargs,
                                   layer_collection=layer_collection,
                                   var_list=self.online_net.weights)
            self.online_net.set_kfac(kfac, self.kfac_inv_upd_interval,
                                     train_weights=self.global_net.weights)
        elif isinstance(self.optimizer, tf.train.Optimizer):
            # if self.optimizer is a (subclass) instance of tf.train.Optimizer
            self.online_net.set_optimizer(self.optimizer,
                                          clip_norm=self.opt_clip_norm,
                                          train_weights=self.global_net.weights)
        else:
            raise ValueError('Optimizer {} invalid'.format(self.optimizer))

    def get_output_dir(self, env_name):
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
            self.print('Made output dir', self.save_dir)
        save_dir = self.save_dir
        experiment_id = 0
        for folder_name in os.listdir(save_dir):
            if not os.path.isdir(os.path.join(save_dir, folder_name)):
                continue
            try:
                folder_name = int(folder_name.split('-run')[-1])
                if folder_name > experiment_id:
                    experiment_id = folder_name
            except:
                pass
        experiment_id += 1
        save_dir = os.path.join(save_dir, env_name)
        save_dir += '-run{}'.format(experiment_id)
        os.makedirs(save_dir, exist_ok=True)
        return save_dir

    def save_model(self, step):
        if self.output is not None:
            filename = os.path.join(self.output, 'model_{}.h5'.format(step))
            self.online_net.save_model(filename)
            self.print('keras model written to {}'.format(filename))

    '''
    Methods subject to overloading
    '''
    def setup_algorithm(self):
        raise NotImplementedError

    def setup_nets(self, worker_dev, rep_dev, env):
        # global net
        with tf.device(rep_dev):
            self.global_net = self.build_net(env, is_global=True)
            if self.is_master and self.verbose:
                self.global_net.model.summary()
            self.step_counter = StepCounter()

        if self.num_parallel > 1:
            # local net
            with tf.device(worker_dev):
                self.online_net = self.build_net(env)
                self.online_net.set_loss(**self.loss_kwargs)
                self.set_online_optimizer()
                self.online_net.set_sync_weights(self.global_net.weights)
                self.step_counter.set_increment()
        else:
            self.online_net = self.global_net
            self.online_net.set_loss(**self.loss_kwargs)
            self.set_online_optimizer()
            self.step_counter.set_increment()

    def build_model(self, state, feature, **kwargs):
        raise NotImplementedError

    def set_session(self, sess):
        for obj in self.global_net, self.online_net, self.step_counter:
            obj.set_session(sess)
        if self.load_model is not None:
            self.global_net.set_sync_weights(self.saved_weights)
            self.global_net.sync()

    def sync_to_global(self):
        if self.num_parallel > 1:
            self.online_net.sync()
        if self.noisynet is not None:
            self.online_net.sample_noise()

    def train_on_batch(self, batch, batch_weight=None):
        rl_state = []
        rl_slice = []
        last_index = 0
        unrolled_batch = []
        for rlist in batch:
            for rollout in rlist:
                unrolled_batch.append(rollout)
                r_state = []
                for state in rollout.state_list:
                    r_state.append(self.state_to_input(state))
                r_state = np.array(r_state)
                rl_state.append(r_state)
                index = last_index + len(r_state)
                rl_slice.append(slice(last_index, index))
                last_index = index
        cc_state = np.concatenate(rl_state)

        if batch_weight is None:
            cc_weight = None
        else:
            cc_weight = [weight for rlist, weight in zip(batch, batch_weight)
                         for rollout in rlist for _ in range(len(rollout))]

        # cc_boots is a tuple of concatenated bootstrap quantities
        cc_boots = self.concat_bootstrap(cc_state, rl_slice)

        # rl_boots is a list of tuple of boostrap quantities
        # and each tuple corresponds to a rollout
        rl_boots = [tuple(boot[r_slice] for boot in cc_boots)
                    for r_slice in rl_slice]

        # feed_list contains all arguments to train_on_batch
        feed_list = []
        for rollout, r_state, r_boot in zip(unrolled_batch, rl_state, rl_boots):
            r_input = r_state[:-1]
            r_feeds = self.rollout_feed(rollout, *r_boot)
            feed_list.append((r_input, *r_feeds))

        # concatenate individual types of feeds from the list
        cc_args = *map(np.concatenate, zip(*feed_list)), cc_weight
        batch_loss, cc_error = self.online_net.train_on_batch(*cc_args)
        batch_error = [np.mean(cc_error[idx:(idx + self.rollout_maxlen)])
                       for idx in range(0, len(cc_error), self.rollout_maxlen)]
        return batch_loss, batch_error

    def concat_bootstrap(self, cc_state):
        raise NotImplementedError

    def rollout_feed(self, rollout, *rollout_bootstraps):
        raise NotImplementedError

    def rollout_target(self, rollout, value_last):
        reward_long = 0.0 if rollout.done else value_last
        r_target = np.zeros(len(rollout))
        for idx in reversed(range(len(rollout))):
            reward_long *= self.discount
            reward_long += rollout.reward_list[idx]
            r_target[idx] = reward_long
        return r_target

