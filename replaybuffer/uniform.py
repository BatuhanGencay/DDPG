""" Vanilla Replay Buffer
"""
from collections import namedtuple
from random import sample as randsample

import numpy as np


class BaseBuffer:
    """ Base class for 1-step buffers. Numpy queue implementation with
    multiple arrays. Sampling efficient in numpy (thanks to fast indexing)

    Arguments:
        - capacity: Maximum size of the buffer
        - state_shape: Shape of a single observation (must be a tuple)
        - state_dtype: Data type of the sumtreestate array. Must be a
        compatible type to numpy
    """

    Transition = namedtuple("Transition",
                            "state action reward next_state terminal")

    def __init__(self, capacity, state_shape, state_dtype):
        self.capacity = capacity
        self.buffer = []

    def __len__(self):
        """ Capacity of the buffer
        """
        return self.capacity

    def push(self, state, action, reward, next_state, done, *args, **kwargs):
        """ Push a transition object (with single elements) to the buffer
        """
        raise NotImplementedError

    def sample(self, batchsize, *args, **kwargs):
        """ Sample a batch of transitions
        """
        raise NotImplementedError


class UniformBuffer(BaseBuffer):
    """ Standard Replay Buffer that uniformly samples the transitions.
    Arguments:
        - capacity: Maximum size of the buffer
        - state_shape: Shape of a single observation (must be a tuple)
        - state_dtype: Data type of the state array. Must be a compatible
        dtype to numpy
    """

    def __init__(self, capacity, state_shape, state_dtype):
        super().__init__(capacity, state_shape, state_dtype)
        self._cycle = 0
        self.size = 0

    def push(self, transition, **kwargs):
        """ Push a transition object (with single elements) to the buffer.
        FIFO implementation using <_cycle>. <_cycle> keeps track of the next
        available index to write. Remember to update <size> attribute as we
        push transitions.
        """

        if self.capacity != len(self.buffer):
            self.buffer.append(transition)
            self.size += 1
        else:
            self.buffer[self._cycle] = transition
            self._cycle = (self._cycle + 1) % self.capacity

    def sample(self, batchsize, *args):
        """ Uniformly sample a batch of transitions from the buffer. If
        batchsize is less than the number of valid transitions in the buffer
        return None. The return value must be a Transition object with batch
        of state, actions, .. etc.
            Return: T(states, actions, rewards, terminals, next_states)
        """
        if batchsize > self.size:
            return None
        batch = randsample(self.buffer, batchsize)

        states, actions, rewards, next_states, terminals = [], [], [], [], []

        try:
            for var in batch:
                states.append(var[0])
                actions.append([var[1]])
                rewards.append([var[2]])
                next_states.append(var[3])
                terminals.append([var[4]])
        except TypeError:
            pass
        return self.Transition(np.asarray(states), np.asarray(actions, dtype='int64'),
                                  np.asarray(rewards, dtype='float32'), np.asarray(next_states),
                                  np.asarray(terminals, dtype='float32'))
