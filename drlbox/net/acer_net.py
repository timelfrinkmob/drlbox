
import tensorflow as tf
import gym.spaces
from .ac_net import ACNet


'''
ACER assumes discrete action for now.
'''
class ACERNet(ACNet):

    @classmethod
    def from_sfa(cls, state, feature, action_space):
        self = cls()
        flatten = tf.keras.layers.Flatten()
        if type(feature) is tuple:
            # separated logits/value streams when feature is a length 2 tuple
            feature_logits, feature_value = map(flatten, feature)
        else:
            # feature is a single stream otherwise
            feature_logits = feature_value = flatten(feature)
        self.action_mode = self.DISCRETE
        size_logits = action_space.n
        init = tf.keras.initializers.RandomNormal(stddev=1e-3)
        logits_layer = self.dense_layer(size_logits, kernel_initializer=init)
        logits = logits_layer(feature_logits)
        value = self.dense_layer(size_logits)(feature_value)
        model = tf.keras.models.Model(inputs=state, outputs=[logits, value])
        self.set_model(model)
        return self

    def set_model(self, model):
        self.model = model
        self.weights = model.weights
        self.ph_state, = model.inputs
        self.tf_logits, self.tf_value = model.outputs

    def set_loss(self, entropy_weight=0.01, kl_weight=0.1, truc_max=10.0):
        num_action = self.tf_logits.shape[1]
        ph_action = tf.placeholder(tf.int32, [None])
        action_onehot = tf.one_hot(ph_action, depth=num_action)

        # importance sampling weight and trunc
        ph_lratio = tf.placeholder(tf.float32, [None, num_action])
        trunc = tf.minimum(truc_max, ph_lratio)
        trunc_act = tf.reduce_sum(trunc * action_onehot, axis=1)

        # return and value placeholders
        ph_q_ret = tf.placeholder(tf.float32, [None])
        ph_q_val = tf.placeholder(tf.float32, [None, num_action])
        ph_baseline = tf.placeholder(tf.float32, [None])

        # log policy
        log_probs = tf.nn.log_softmax(self.tf_logits)

        # policy loss: sampled return
        log_probs_act = tf.reduce_sum(log_probs * action_onehot, axis=1)
        adv_ret = ph_q_ret - ph_baseline
        policy_ret_loss = -tf.reduce_sum(trunc_act * log_probs_act * adv_ret)

        # policy loss: bootstrapped value
        probs = tf.nn.softmax(self.tf_logits)
        probs_const = tf.stop_gradient(probs)
        tru_prob = tf.maximum(0.0, 1.0 - truc_max / ph_lratio) * probs_const
        adv_val = ph_q_val - ph_baseline[:, tf.newaxis]
        policy_val_loss = -tf.reduce_sum(tru_prob * log_probs * adv_val)

        # KL (wrt averaged policy net) loss
        ph_avg_logits = tf.placeholder(tf.float32, [None, num_action])
        avg_probs = tf.nn.softmax(ph_avg_logits)
        kl_loss = -kl_weight * tf.reduce_sum(avg_probs * log_probs)

        # value (critic) loss
        value_act = tf.reduce_sum(self.tf_value * action_onehot, axis=1)
        value_squared_diff = tf.squared_difference(ph_q_ret, value_act)
        value_loss = tf.reduce_sum(value_squared_diff)

        # total loss
        self.tf_loss = policy_ret_loss + policy_val_loss + value_loss + kl_loss

        # entropy
        if entropy_weight:
            self.tf_loss += tf.reduce_sum(probs * log_probs) * entropy_weight

        # placeholders
        self.ph_action = ph_action
        self.ph_lratio = ph_lratio
        self.ph_q_ret = ph_q_ret
        self.ph_q_val = ph_q_val
        self.ph_baseline = ph_baseline
        self.ph_avg_logits = ph_avg_logits

    def ac_values(self, state):
        return self.sess.run([self.tf_logits, self.tf_value],
                             feed_dict={self.ph_state: state})

    def train_on_batch(self, state, action, lratio, q_ret, q_val,
                       baseline, avg_logits):
        feed_dict = {self.ph_state:         state,
                     self.ph_action:        action,
                     self.ph_lratio:        lratio,
                     self.ph_q_ret:         q_ret,
                     self.ph_q_val:         q_val,
                     self.ph_baseline:      baseline,
                     self.ph_avg_logits:    avg_logits}
        loss = self.sess.run(self.op_train, feed_dict=feed_dict)[0]
        return loss

    def set_soft_update(self, new_weights, update_ratio):
        assign_list = []
        for wt, nwt in zip(self.weights, new_weights):
            upd = (1.0 - update_ratio) * wt + update_ratio * nwt
            assign_list.append(wt.assign(upd))
        self.op_soft_update = tf.group(*assign_list)

    def soft_update(self):
        self.sess.run(self.op_soft_update)

