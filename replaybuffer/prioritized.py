import numpy as np

from replaybuffer.uniform import BaseBuffer, UniformBuffer
from replaybuffer.seg_tree import SumTree, MinTree


class PriorityBuffer(BaseBuffer):
    """ Prioritized Replay Buffer that sample transitions with a probability
    that is proportional to their respected priorities.
        Arguments:
            - capacity: Maximum size of the buffer
            - state_shape: Shape of a single observation (must be a tuple)
            - state_dtype: Data type of the state array. Must be a compatible
            dtype to numpy
    """

    def __init__(self, capacity, state_shape, state_dtype,
                 alpha, epsilon=0.1):
        super().__init__(capacity, state_shape, state_dtype)
        self.sumtree = SumTree(capacity)
        # self.mintree = MinTree(capacity)
        self.epsilon = epsilon
        self.alpha = alpha
        self._cycle = 0
        self.size = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

        self.max_p = epsilon ** alpha

    def push(self, state, action, reward, next_state, done, priority, **kwargs):
        """ Push a transition object (with single elements) to the buffer.
        Actors push with own calculated priority
        """

        transition = UniformBuffer.Transition(state, action, reward, next_state, done)

        if self.capacity != len(self.buffer):
            self.buffer.append(transition)
            self.sumtree.push(priority)
            self.size += 1
        else:
            self.buffer[self._cycle] = transition
            self.sumtree.update(self._cycle, priority)
            self._cycle = (self._cycle + 1) % self.capacity

    def _get_priority(self, error):
        return (np.abs(error) + self.epsilon) ** self.alpha

    def sample(self, batch_size, beta, **kwargs):
        """ Sample a transition based on priorities.
            Arguments:
                - batch_size: Size of the batch
                - beta: Importance sampling weighting annealing
                        To what degree to use importance weights
                        (0 - no corrections, 1 - full correction)
            Return:
                - batch of samples
                - indexes of the sampled transitions (so that corresponding
                priorities can be updated)
                - Importance sampling weights
        """

        if batch_size > self.size:
            return None

        tree_indices = []
        data_indices = []
        priorities = []
        states, actions, rewards, next_states, terminals = [], [], [], [], []

        segment_size = self.sumtree.total() / batch_size
        for i in range(batch_size):
            choice = np.random.uniform(i * segment_size, (i + 1) * segment_size)
            tree_index = self.sumtree.get(choice)
            data_index = tree_index - self.capacity + 1
            tree_indices.append(tree_index)
            data_indices.append(data_index)
            priorities.append(self.sumtree.get_priority(tree_index))
            transition = self.buffer[data_index]
            states.append(transition[0])
            actions.append([transition[1]])
            rewards.append([transition[2]])
            next_states.append(transition[3])
            terminals.append([transition[4]])

        samples = self.Transition(states, actions,
                                  rewards, next_states,
                                  terminals)

        # probabilities across the samples
        probs = priorities / self.sumtree.total()

        N = len(self.buffer)
        weights = (N * probs) ** (-beta)
        weights /= weights.max()

        return samples, np.asarray(data_indices), np.asarray(weights)

    def update_priority(self, indexes, values):
        """ Update the priority values of given indexes (for both min and sum
        trees). Remember to update max_p value! """

        for idx, val in zip(indexes, values):
            self.sumtree.update(idx, (val + self.epsilon) ** self.alpha)

