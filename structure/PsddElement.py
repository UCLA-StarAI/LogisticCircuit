from structure.PsddNode import PsddNode
import random

class PsddElement(object):
    """AND gate.
    In this implementation, we assume every AND gate is the child of one PSDD decision nodes (OR gate).
    In another words, they are not shared between different PSDD decision nodes."""

    def __init__(self, prime: PsddNode, sub: PsddNode):
        self._prime = prime
        self._sub = sub
        self._prime.increase_num_parents_by_one()
        self._sub.increase_num_parents_by_one()
        # difference between prob and feature:
        # prob is calculated in a bottom-up pass and only considers values of variables the element has
        # feature is calculated in a top-down pass using probs; equals the WMC of that element reached
        self._feature = None
        self._prob = None
        self._parameter = random.random() - 0.5
        self._parent = None
        self._splittable_variables = set()
        self._flag = False

    @property
    def prime(self):
        return self._prime

    @prime.setter
    def prime(self, value):
        self._prime = value
        if self._prime is not None:
            self._prime.increase_num_parents_by_one()
    
    @property
    def sub(self):
        return self._sub

    @sub.setter
    def sub(self, value):
        self._sub = value
        if self._sub is not None:
            self._sub.increase_num_parents_by_one()

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

    def calculate_prob(self):
        self._prob = self._prime.prob + self._sub.prob

    @property
    def parameter(self):
        return self._parameter

    @parameter.setter
    def parameter(self, value):
        self._parameter = value

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, value):
        self._parent = value

    @property
    def splittable_variables(self):
        return self._splittable_variables

    @splittable_variables.setter
    def splittable_variables(self, value):
        self._splittable_variables = value

    def remove_splittable_variable(self, variable_to_remove):
        self._splittable_variables.discard(variable_to_remove)

    @property
    def flag(self):
        return self._flag

    @flag.setter
    def flag(self, value):
        self._flag = value
