""" Sum Tree implementation for the prioritized
replay buffer.
"""

import numpy as np


class SegTree:
    """ Base Segment Tree Class with binary heap implementation that push
    values as a Queue(FIFO).
        Arguments:
            - capacity: Maximum size of the tree. It also the number of leaf
            nodes in the tree.
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self._cycle = 0
        self.size = 0

        self.tree = np.zeros((self.capacity * 2,))
        self.tree = np.delete(self.tree, 0)

    def push(self, value):
        """ Push a value into the tree by calling the update method. Push
        function overrides values when the tree is full  """

        # self.tree[self._cycle + self.capacity] = value
        self.update(self._cycle, value)
        self._cycle = (self._cycle + 1) % self.capacity
        self.size += 1

    def update(self, index, value):
        # adding first (ignored) index to the list to get right result
        self.tree = np.append(0, self.tree)

        index += self.capacity
        self.tree[index] = value

        while index > 1:
            index = int(index / 2)
            i = index * 2
            self.tree[index] = min(self.tree[i], self.tree[i + 1])

        # getting rid of the first (ignored) index again
        self.tree = np.delete(self.tree, 0)


class SumTree(SegTree):
    """ A Binary tree with the property that a parent node is the sum of its
    two children.
        Arguments:
            - capacity: Maximum size of the tree. It also the number of leaf
            nodes in the tree.
    """

    def __init__(self, capacity):
        super().__init__(capacity)

    def get(self, value):
        """ Return the index (ranging from 0 to max capacity) that corresponds
        to the given value """

        if value > self.tree[0]:
            raise ValueError("Value is greater than the root")

        index = 0

        while True:
            l_idx = index * 2 + 1
            r_idx = l_idx + 1
            if (l_idx) >= len(self.tree):
                break
            if value <= self.tree[l_idx]:
                index = l_idx
            else:
                value -= self.tree[l_idx]
                index = r_idx

        return index

    def update_new(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        while tree_idx != 0:  
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def update(self, index, value):
        """ Update the value of the given index (ranging from 0 to max
        capacity) with the given value
        """
        self.tree = np.append(0, self.tree)

        index += self.capacity
        self.tree[index] = value

        while index > 1:
            index = int(index / 2)
            i = index * 2
            self.tree[index] = self.tree[i] + self.tree[i + 1]

        # getting rid of the first (ignored) index again
        self.tree = np.delete(self.tree, 0)

    def get_max(self):
        return max(self.tree[(len(self.tree)) // 2:])

    def total(self):
        return self.tree[0]

    def get_priority(self, index):
        return self.tree[index]


class MinTree(SegTree):
    """ A Binary tree with the property that a parent node is the minimum of
    its two children.
        Arguments:
            - capacity: Maximum size of the tree. It also the number of leaf
            nodes in the tree.
    """

    def __init__(self, capacity):
        super().__init__(capacity)
        self.tree[:] = np.inf

    def update(self, index, value):
        """ Update the value of the given index (ranging from 0 to max
        capacity) with the given value
        """
        assert value >= 0, "Value cannot be negative"

        # adding first (ignored) index to the list to get right result
        self.tree = np.append(0, self.tree)

        index += self.capacity
        self.tree[index] = value

        while index > 1:
            index = int(index / 2)
            i = index * 2
            self.tree[index] = min(self.tree[i], self.tree[i + 1])

        self.tree = np.delete(self.tree, 0)

    @property
    def minimum(self):
        """ Return the minimum value of the tree (root node). Complexity: O(1)
        """
        return self.tree[0]
