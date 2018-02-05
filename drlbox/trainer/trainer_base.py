
from multiprocessing import Process, cpu_count
import socket
from .blocker import Blocker

import os
import signal
import tensorflow as tf
import builtins
from numpy import concatenate
from drlbox.common.util import set_args
from .step_counter import StepCounter


print = lambda *args, **kwargs: builtins.print(*args, **kwargs, flush=True)

LOCALHOST = 'localhost'
JOBNAME = 'local'

class Trainer:

    KEYWORD_DICT = dict(env_maker=None,
                        feature_maker=None,
                        state_to_input=None,
                        load_model=None,
                        save_dir=None,              # Directory to save data to
                        num_parallel=cpu_count(),
                        port_begin=2220,
                        discount=0.99,
                        train_steps=1000000,
                        opt_learning_rate=1e-4,
                        opt_batch_size=32,
                        opt_adam_epsilon=1e-4,
                        opt_grad_clip_norm=40.0,
                        interval_sync_target=40000,
                        interval_save=10000,)

    need_target_net = False

    def __init__(self, **kwargs):
        set_args(self, self.KEYWORD_DICT, kwargs)

    def run(self):
        self.port_list = [self.port_begin + i for i in range(self.num_parallel)]
        for port in self.port_list:
            if not self.port_available(LOCALHOST, port):
                raise NameError('port {} is not available'.format(port))
        print('Claiming {} port {} ...'.format(LOCALHOST, self.port_list))
        worker_list = []
        for wid in range(self.num_parallel):
            worker = Process(target=self.worker, args=(wid,))
            worker.start()
            worker_list.append(worker)
        Blocker().block()
        for worker in worker_list:
            worker.terminate()
        print('Asynchronous training has ended')

    def worker(self, wid):
        env = self.env_maker()
        self.output = self.get_output_dir(env.spec.id)

        # ports, cluster, and server
        self.is_master = wid == 0
        cluster_list = ['{}:{}'.format(LOCALHOST, p) for p in self.port_list]
        cluster = tf.train.ClusterSpec({JOBNAME: cluster_list})
        server = tf.train.Server(cluster, job_name=JOBNAME, task_index=wid)
        print('Starting server #{}'.format(wid))

        # global/local devices
        worker_dev = '/job:{}/task:{}/cpu:0'.format(JOBNAME, wid)
        rep_dev = tf.train.replica_device_setter(worker_device=worker_dev,
                                                 cluster=cluster)

        self.setup_algorithm(env.action_space)

        # global net
        with tf.device(rep_dev):
            if self.load_model is None:
                global_net = self.build_net(env)
            else:
                saved_model = self.net_cls.load_model(self.load_model)
                saved_weights = saved_model.get_weights()
                global_net = self.net_cls.from_model(saved_model)
            if self.is_master:
                global_net.model.summary()
            self.step_counter = StepCounter()

        # local net
        with tf.device(worker_dev):
            self.online_net = self.build_net(env)
            self.online_net.set_loss(**self.loss_kwargs)
            self.online_net.set_optimizer(**self.opt_kwargs,
                                     train_weights=global_net.weights)
            self.online_net.set_sync_weights(global_net.weights)
            self.step_counter.set_increment()

        # build a separate global target net for dqn
        if self.need_target_net:
            with tf.device(rep_dev):
                self.target_net = self.build_net(env)
                self.target_net.set_sync_weights(global_net.weights)
        else: # make target net a reference to the local net
            self.target_net = self.online_net

        # begin tensorflow session, build async RL agent and train
        port = self.port_list[wid]
        with tf.Session('grpc://{}:{}'.format(LOCALHOST, port)) as sess:
            sess.run(tf.global_variables_initializer())
            for obj in global_net, self.online_net, self.step_counter:
                obj.set_session(sess)
            if self.target_net is not self.online_net:
                self.target_net.set_session(sess)
            if self.load_model is not None:
                global_net.set_sync_weights(saved_weights)
                global_net.sync()

            # train the agent
            self.train_on_env(env)

            # terminates the entire training when the master worker terminates
            if self.is_master:
                print('Master worker terminates -- sending SIGTERM to parent')
                os.kill(os.getppid(), signal.SIGTERM)

    def train_on_env(self, env):
        step = self.step_counter.step_count()
        if self.is_master:
            last_save = step
            last_sync_target = step
            self.save_model(step)

        state = env.reset()
        state = self.state_to_input(state)
        episode_reward = 0.0
        while step <= self.train_steps:
            self.online_net.sync()
            rollout_list = [self.rollout_builder(state)]
            for batch_step in range(self.opt_batch_size):
                action_values = self.online_net.action_values([state])[0]
                action = self.policy.select_action(action_values)
                state, reward, done, info = env.step(action)
                episode_reward += reward
                state = self.state_to_input(state)
                rollout_list[-1].append(state, action, reward, done)
                if done:
                    state = env.reset()
                    state = self.state_to_input(state)
                    if batch_step < self.opt_batch_size - 1:
                        rollout_list.append(self.rollout_builder(state))
                    print('episode reward {:5.2f}'.format(episode_reward))
                    episode_reward = 0.0

            '''
            feed_list is a list of tuples:
            (inputs, actions, advantages, targets) for actor-critic;
            (inputs, targets) for dqn.
            '''
            feed_list = [rollout.get_feed(self.target_net, self.online_net)
                         for rollout in rollout_list]

            # concatenate individual types of feeds from the list
            train_args = map(concatenate, zip(*feed_list))
            batch_loss = self.online_net.train_on_batch(*train_args)

            self.step_counter.increment(self.opt_batch_size)
            step = self.step_counter.step_count()
            if self.target_net is not self.online_net:
                if step - last_sync_target > self.interval_sync_target:
                    self.target_net.sync()
                    last_sync_target = step
            if self.is_master:
                if step - last_save > self.interval_save:
                    self.save_model(step)
                    last_save = step
                str_step = 'training step {}/{}'.format(step, self.train_steps)
                print(str_step + ', loss {:3.3f}'.format(batch_loss))
        # save at the end of training
        if self.is_master:
            self.save_model(step)

    def port_available(self, host, port):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        return sock.connect_ex((host, port)) != 0

    def setup_algorithm(self, action_space):
        raise NotImplementedError

    def build_net(self, env):
        state, feature = self.feature_maker(env.observation_space)
        return self.net_cls.from_sfa(state, feature, env.action_space)

    def get_output_dir(self, env_name):
        if self.save_dir is None:
            return None
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
            print('Made output dir', self.save_dir)
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
            print('keras model written to {}'.format(filename))
