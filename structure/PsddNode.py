from structure.Vtree import Vtree
import numpy as np
import random

class PsddNode(object):

    def __init__(self, vtree: Vtree, index):
        self._vtree = vtree
        self._index = index
        self._splittable_variables = None
        self._num_parents = 0
        # difference between prob and feature:
        # prob is calculated in a bottom-up pass and only considers values of variables the node has
        # feature is calculated in a top-down pass using probs; equals the WMC of that node reached
        self._prob = None
        self._feature = None

    @property
    def vtree(self):
        return self._vtree

    @property
    def index(self):
        return self._index

    @property
    def num_parents(self):
        return self._num_parents

    def increase_num_parents_by_one(self):
        self._num_parents += 1

    def decrease_num_parents_by_one(self):
        self._num_parents -= 1
        
    @property
    def feature(self):
        return self._feature

    @feature.setter
    def feature(self, value):
        self._feature = value

    @property
    def prob(self):
        return self._prob

    @prob.setter
    def prob(self, value):
        self._prob = value


class PsddDecision(PsddNode):
    """OR gate."""

    def __init__(self, vtree, index, elements: list):
        super().__init__(vtree, index)
        self._elements = elements
        for element in elements:
            element.parent = self

    @property
    def elements(self):
        return self._elements

    def add_element(self, element):
        self._elements.append(element)
        element.parent = self

    def remove_element(self, index):
        del self._elements[index]

    def calculate_prob(self):
        if len(self._elements) == 0:
            raise ValueError("Decision nodes should have at least one elements.")
        for element in self._elements:
            element.calculate_prob()
        self._prob = np.sum([np.exp(element.prob) for element in self._elements], axis=0)
        self._prob[np.where(self._prob < 1e-7)[0]] = 1e-7
        self._prob = np.log(self._prob)
        for element in self._elements:
            element.prob -= self._prob
        self._prob[np.where(self._prob > 0.0)[0]] = 0.0
        self._feature = np.zeros(shape=self._prob.shape, dtype=np.float32)

    def calculate_feature(self):
        feature = np.log(self._feature)
        for element in self._elements:
            element.feature = np.exp(feature + element.prob)
            element.prime.feature += element.feature
            element.sub.feature += element.feature


LITERAL_UNSATISFIABLE = -1
LITERAL_ALWAYS_SATISFIED = 2
LITERAL_IS_TRUE = 1
LITERAL_IS_FALSE = 0


class PsddTerminal(PsddNode):
    """Terminal(leaf) node."""

    def __init__(self, vtree, index, var_index, var_value):
        super().__init__(vtree, index)
        self._var_index = var_index
        self._var_value = var_value
        self._parameter = random.random() - 0.5

    @property
    def var_index(self):
        return self._var_index

    @var_index.setter
    def var_index(self, value):
        self._var_index = value

    @property
    def var_value(self):
        return self._var_value

    @var_value.setter
    def var_value(self, value):
        self._var_value = value

    @property
    def parameter(self):
        return self._parameter

    @parameter.setter
    def parameter(self, value):
        self._parameter = value

    def calculate_prob(self, samples: np.array):
        if self._var_value == LITERAL_IS_TRUE:
            self._prob = np.log(samples[:, self._var_index - 1])
        elif self._var_value == LITERAL_IS_FALSE:
            self._prob = np.log(1.0 - samples[:, self._var_index - 1])
        else:
            raise ValueError('Terminal nodes should either be positive literals or negative literals.')
        self._feature = np.zeros(shape=self._prob.shape, dtype=np.float32)

