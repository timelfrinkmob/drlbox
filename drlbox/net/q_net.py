
import tensorflow as tf
from .net_base import RLNet


class QNet(RLNet):

    def set_model(self, model):
        self.model = model
        self.weights = model.weights
        self.ph_state, = model.inputs
        self.tf_values, = model.outputs

    def set_loss(self):
        ph_action = tf.placeholder(tf.int32, [None])
        action_onehot = tf.one_hot(ph_action, depth=self.tf_values.shape[1])
        ph_target = tf.placeholder(tf.float32, [None])
        act_values = tf.reduce_sum(self.tf_values * action_onehot, axis=1)
        self.tf_loss = tf.losses.huber_loss(ph_target, act_values)
        kfac_value_loss = 'normal_predictive', (self.tf_values,)
        self.kfac_loss_list = [kfac_value_loss]
        self.ph_train_list = [self.ph_state, ph_action, ph_target]

    def action_values(self, state):
        return self.sess.run(self.tf_values, feed_dict={self.ph_state: state})

