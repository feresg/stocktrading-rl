import tensorflow as tf
import keras.backend as K

from keras.initializers import RandomUniform
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Dense, Reshape, Lambda, BatchNormalization, GaussianNoise, Flatten, concatenate

from tqdm import tqdm
import numpy as np
from collections import deque


class Actor:
    """Actor Network for the DDPG Algorithm"""

    def __init__(self, inp_dim, out_dim, act_range, lr, tau):
        self.env_dim = inp_dim
        self.act_dim = out_dim
        self.act_range = act_range
        self.tau = tau
        self.lr = lr
        self.model = self.network()
        self.target_model = self.network()
        self.adam_optimizer = self.optimizer()

    def network(self):
        """ Actor Network for Policy function Approximation, using a tanh
        activation for continuous control. We add parameter noise to encourage
        exploration, and balance it with Layer Normalization.
        """
        inp = Input((self.env_dim))
        #
        x = Dense(256, activation='relu')(inp)
        x = GaussianNoise(1.0)(x)
        #
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = GaussianNoise(1.0)(x)
        #
        out = Dense(self.act_dim, activation='tanh',
                    kernel_initializer=RandomUniform())(x)
        out = Lambda(lambda i: i * self.act_range /
                     2 + self.act_range / 2)(out)
        return Model(inp, out)

    def predict(self, state):
        """ Action prediction"""
        return self.model.predict(np.expand_dims(state, axis=0))

    def target_predict(self, inp):
        """ Action prediction (target network)"""
        return self.target_model.predict(inp)

    def transfer_weights(self):
        """ Transfer model weights to target model with a factor of Tau"""
        W, target_W = self.model.get_weights(), self.target_model.get_weights()
        for i in range(len(W)):
            target_W[i] = self.tau * W[i] + (1 - self.tau) * target_W[i]
        self.target_model.set_weights(target_W)

    def train(self, states, actions, grads):
        """ Actor Training"""
        self.adam_optimizer([states, grads])

    def optimizer(self):
        """ Actor Optimizer"""
        action_gdts = K.placeholder(shape=(None, self.act_dim))
        print(action_gdts)
        params_grad = tf.gradients(
            self.model.output, self.model.trainable_weights, -action_gdts)
        grads = zip(params_grad, self.model.trainable_weights)
        return K.function(inputs=[self.model.input, action_gdts],
                          outputs=[],
                          updates=[tf.train.AdamOptimizer(self.lr).apply_gradients(grads)])

    def save(self, path):
        self.model.save_weights(path + '_actor.h5')

    def load_weights(self, path):
        self.model.load_weights(path)


class Critic:
    """Critic for the DDPG Algorithm, Q-Value function approximator"""

    def __init__(self, inp_dim, out_dim, lr, tau):
        # Dimensions and Hyperparams
        self.env_dim = inp_dim
        self.act_dim = out_dim
        self.tau, self.lr = tau, lr
        # Build models and target models
        self.model = self.network()
        self.target_model = self.network()
        self.model.compile(Adam(self.lr), 'mse')
        self.target_model.compile(Adam(self.lr), 'mse')
        # Function to compute Q-value gradients (Actor Optimization)
        self.action_grads = K.function([self.model.input[0], self.model.input[1]], K.gradients(
            self.model.output, [self.model.input[1]]))

    def network(self):
        """Assemble Critic network to predict q-values"""
        state = Input((self.env_dim))
        action = Input((self.act_dim,))
        x = Dense(256, activation='relu')(state)
        x = concatenate([Flatten()(x), action])
        x = Dense(128, activation='relu')(x)
        out = Dense(1, activation='linear',
                    kernel_initializer=RandomUniform())(x)
        return Model([state, action], out)

    def gradients(self, states, actions):
        """Compute Q-value gradients w.r.t. states and policy-actions"""
        return self.action_grads([states, actions])

    def target_predict(self, inp):
        """Predict Q-Values using the target network"""
        return self.target_model.predict(inp)

    def train_on_batch(self, states, actions, critic_target):
        """Train the critic network on batch of sampled experience"""
        return self.model.train_on_batch([states, actions], critic_target)

    def transfer_weights(self):
        """Transfer model weights to target model with a factor of Tau"""
        W, target_W = self.model.get_weights(), self.target_model.get_weights()
        for i in range(len(W)):
            target_W[i] = self.tau * W[i] + (1 - self.tau) * target_W[i]
        self.target_model.set_weights(target_W)

    def save(self, path):
        self.model.save_weights(path + '_critic.h5')

    def load_weights(self, path):
        self.model.load_weights(path)
