import numpy as np
from collections import deque
import random

import tensorflow as tf


def gather_stats(agent, env):
    """ Compute average rewards over 10 episodes"""
    score = []
    for _ in range(10):
        old_state = env.reset()
        cumul_r, done = 0, False
        while not done:
            a = agent.policy_action(old_state)
            old_state, r, done, _ = env.step(a)
            cumul_r += r
        score.append(cumul_r)
    return np.mean(np.array(score)), np.std(np.array(score))


def tf_summary(tag, val):
    """Scalar Value Tensorflow Summary"""
    return tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=val)])


class MemoryBuffer(object):
    """ Memory Buffer Helper class for Experience Replay using a double-ended queue"""

    def __init__(self, buffer_size):
        """ Initialization"""
        self.buffer = deque()
        self.count = 0
        self.buffer_size = buffer_size

    def memorize(self, state, action, reward, done, new_state, error=None):
        """ Save an experience to memory, optionally with its TD-Error"""

        experience = (state, action, reward, done, new_state)
        # Check if buffer is already full
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        """ Current Buffer Occupation"""
        return self.count

    def sample_batch(self, batch_size):
        """ Sample a batch"""
        batch = []

        # Sample randomly from Buffer
        if self.count < batch_size:
            idx = None
            batch = random.sample(self.buffer, self.count)
        else:
            idx = None
            batch = random.sample(self.buffer, batch_size)

        # Return a batch of experience
        s_batch = np.array([i[0] for i in batch])
        a_batch = np.array([i[1] for i in batch])
        r_batch = np.array([i[2] for i in batch])
        d_batch = np.array([i[3] for i in batch])
        new_s_batch = np.array([i[4] for i in batch])
        return s_batch, a_batch, r_batch, d_batch, new_s_batch, idx

    def clear(self):
        """ Clear buffer / Sum Tree"""
        self.buffer = deque()
        self.count = 0


class OrnsteinUhlenbeckProcess(object):
    """Ornstein-Uhlenbeck Noise"""

    def __init__(self, theta=0.15, mu=0, sigma=1, x0=0, dt=1e-2, n_steps_annealing=100, size=1):
        self.theta = theta
        self.sigma = sigma
        self.n_steps_annealing = n_steps_annealing
        self.sigma_step = - self.sigma / float(self.n_steps_annealing)
        self.x0 = x0
        self.mu = mu
        self.dt = dt
        self.size = size

    def generate(self, step):
        sigma = max(0, self.sigma_step * step + self.sigma)
        x = self.x0 + self.theta * (self.mu - self.x0) * self.dt + \
            sigma * np.sqrt(self.dt) * np.random.normal(size=self.size)
        self.x0 = x
        return x
