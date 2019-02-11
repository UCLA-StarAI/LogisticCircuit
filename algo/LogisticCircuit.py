from structure.CircuitNode import *
from structure.AndGate import AndGate
from structure.Vtree import Vtree
from collections import deque
import numpy as np
from algo.LogisticRegression import LogisticRegression
import random
import copy

class LogisticCircuit(object):

    def __init__(self, vtree: Vtree):
        self._num_variables = vtree.var_count
        self._terminal_nodes = [None] * 2 * self._num_variables
        self._num_created_nodes = 0
        self._generate_all_terminal_nodes(vtree)
        self._root = self._new_logistic_psdd(vtree)
        self._decision_nodes = None
        self._elements = None
        self._num_parameters = None
        self._parameters = None
        self._bias = random.random() - 0.5
        self._serialize()

    @property
    def num_parameters(self):
        return self._num_parameters

    @property
    def parameters(self):
        return self._parameters

    def _generate_all_terminal_nodes(self, vtree: Vtree):
        if vtree.is_leaf():
            var_index = vtree.var
            self._terminal_nodes[var_index - 1] = \
                CircuitTerminal(vtree, self._num_created_nodes, var_index, LITERAL_IS_TRUE)
            self._num_created_nodes += 1
            self._terminal_nodes[self._num_variables + var_index - 1] = \
                CircuitTerminal(vtree, self._num_created_nodes, var_index, LITERAL_IS_FALSE)
            self._num_created_nodes += 1
        else:
            self._generate_all_terminal_nodes(vtree.left)
            self._generate_all_terminal_nodes(vtree.right)

    def _new_logistic_psdd(self, vtree) -> CircuitNode:
        left_vtree = vtree.left
        right_vtree = vtree.right
        prime_variable = left_vtree.var
        sub_variable = right_vtree.var
        elements = list()
        if left_vtree.is_leaf() and right_vtree.is_leaf():
            elements.append(AndGate(self._terminal_nodes[prime_variable - 1],
                                    self._terminal_nodes[sub_variable - 1]))
            elements.append(AndGate(self._terminal_nodes[prime_variable - 1],
                                    self._terminal_nodes[self._num_variables + sub_variable - 1]))
            elements.append(AndGate(self._terminal_nodes[self._num_variables + prime_variable - 1],
                                    self._terminal_nodes[sub_variable - 1]))
            elements.append(AndGate(self._terminal_nodes[self._num_variables + prime_variable - 1],
                                    self._terminal_nodes[self._num_variables + sub_variable - 1]))
        elif left_vtree.is_leaf():
            elements.append(AndGate(self._terminal_nodes[prime_variable - 1],
                                    self._new_logistic_psdd(right_vtree)))
            elements.append(AndGate(self._terminal_nodes[self._num_variables + prime_variable - 1],
                                    self._new_logistic_psdd(right_vtree)))
            for element in elements:
                element.splittable_variables = copy.deepcopy(right_vtree.variables)
        elif right_vtree.is_leaf():
            elements.append(AndGate(self._new_logistic_psdd(left_vtree),
                                    self._terminal_nodes[sub_variable - 1]))
            elements.append(AndGate(self._new_logistic_psdd(left_vtree),
                                    self._terminal_nodes[self._num_variables + sub_variable - 1]))
            for element in elements:
                element.splittable_variables = copy.deepcopy(left_vtree.variables)
        else:
            elements.append(AndGate(self._new_logistic_psdd(left_vtree),
                                    self._new_logistic_psdd(right_vtree)))
            elements[0].splittable_variables = copy.deepcopy(vtree.variables)
        root = OrGate(vtree, self._num_created_nodes, elements)
        self._num_created_nodes += 1
        return root

    def _serialize(self):
        """Serialize all the decision nodes in the logistic psdd.
           Serialize all the elements in the logistic psdd. """
        self._decision_nodes = [self._root]
        self._elements = []
        decision_node_indices = set()
        decision_node_indices.add(self._root.index)
        unvisited = deque()
        unvisited.append(self._root)
        self._num_parameters = 0
        while len(unvisited) > 0:
            current = unvisited.popleft()
            for element in current.elements:
                self._elements.append(element)
                element.flag = False
                if isinstance(element.prime, OrGate) and element.prime.index not in decision_node_indices:
                    decision_node_indices.add(element.prime.index)
                    self._decision_nodes.append(element.prime)
                    unvisited.append(element.prime)
                if isinstance(element.sub, OrGate) and element.sub.index not in decision_node_indices:
                    decision_node_indices.add(element.sub.index)
                    self._decision_nodes.append(element.sub)
                    unvisited.append(element.sub)
        self._parameters = [self._bias]
        self._parameters.extend([terminal_node.parameter for terminal_node in self._terminal_nodes])
        self._parameters.extend([element.parameter for element in self._elements])
        self._parameters = np.array(self._parameters).astype(np.float32)
        self._num_parameters = len(self._parameters)

    def _record_learned_parameters(self, parameters):
        self._parameters = parameters
        self.bias = self._parameters[0].item()
        for i in range(len(self._terminal_nodes)):
            self._terminal_nodes[i].parameter = self._parameters[i + 1].item()
        for i in range(len(self._elements)):
            self._elements[i].parameter = self._parameters[i + 1 + 2*self._num_variables].item()

    def calculate_features(self, images: np.array):
        num_images = images.shape[0]
        for terminal_node in self._terminal_nodes:
            terminal_node.calculate_prob(images)
        for decision_node in reversed(self._decision_nodes):
            decision_node.calculate_prob()
        self._root.feature = np.ones(shape=(num_images,), dtype=np.float32)
        for decision_node in self._decision_nodes:
            decision_node.calculate_feature()
        # bias feature
        bias_features = np.ones(shape=(num_images,), dtype=np.float32)
        terminal_node_features = np.vstack([terminal_node.feature for terminal_node in self._terminal_nodes])
        element_features = np.vstack([element.feature for element in self._elements])
        features = np.vstack((bias_features, terminal_node_features, element_features))
        for terminal_node in self._terminal_nodes:
            terminal_node.feature = None
            terminal_node.prob = None
        for element in self._elements:
            element.feature = None
            element.prob = None
        return features.T

    def _select_element_and_variable_to_split(self, images, features, labels, num_splits):
        y = self.predict(features)
        y = y.reshape(len(features), 1)
        element_gradients = ((labels - y) * features)[:, 2*self._num_variables + 1:]
        element_gradient_variance = np.var(element_gradients, axis=0)
        candidates = sorted(zip(self._elements, element_gradient_variance, features.T[2 * self._num_variables + 1:]),
                            reverse=True, key=lambda x: x[1])
        selected, i = [], 0
        while len(selected) < num_splits and i < len(candidates):
            candidate = candidates[i]
            i += 1
            element_to_split = candidate[0]
            if len(element_to_split.splittable_variables) > 0:
                original_feature = candidate[2]
                original_variance = candidate[1]
                variable_to_split = None
                min_after_split_variance = float('inf')
                for variable in element_to_split.splittable_variables:
                    left_feature = original_feature * images[:, variable - 1]
                    right_feature = original_feature - left_feature
                    left_gradient = (labels - y).T * left_feature
                    right_gradient = (labels - y).T * right_feature
                    w = np.sum(images[:, variable - 1]) / images.shape[0]
                    after_split_variance = w * np.var(left_gradient) + (1 - w) * np.var(right_gradient)
                    if after_split_variance < min_after_split_variance:
                        min_after_split_variance = after_split_variance
                        variable_to_split = variable
                if min_after_split_variance < original_variance:
                    selected.append((element_to_split, variable_to_split))
        return selected

    def _split(self, element_to_split, variable_to_split, depth):
        parent = element_to_split.parent
        original_element, copied_element = self._copy_and_modify_element_for_split(element_to_split, variable_to_split,
                                                                                   0, depth)
        if original_element is None or copied_element is None:
            raise ValueError("Split elements become invalid.")
        parent.add_element(copied_element)

    def _copy_and_modify_element_for_split(self, original_element, variable, current_depth, max_depth):
        original_element.flag = True
        original_element.remove_splittable_variable(variable)
        original_prime = original_element.prime
        original_sub = original_element.sub
        if current_depth >= max_depth:
            if variable in original_prime.vtree.variables:
                original_prime, copied_prime = \
                    self._copy_and_modify_node_for_split(original_prime, variable, current_depth, max_depth)
                copied_sub = original_sub
            elif variable in original_sub.vtree.variables:
                original_sub, copied_sub = \
                    self._copy_and_modify_node_for_split(original_sub, variable, current_depth, max_depth)
                copied_prime = original_prime
            else:
                copied_prime = original_prime
                copied_sub = original_sub
        else:
            original_prime, copied_prime = \
                self._copy_and_modify_node_for_split(original_prime, variable, current_depth, max_depth)
            original_sub, copied_sub = \
                self._copy_and_modify_node_for_split(original_sub, variable, current_depth, max_depth)
        if copied_prime is not None and copied_sub is not None:
            copied_element = AndGate(copied_prime, copied_sub)
            copied_element.parameter = original_element.parameter
            copied_element.splittable_variables = copy.deepcopy(original_element.splittable_variables)
        else:
            copied_element = None
        if original_prime is not None and original_sub is not None:
            original_element.prime = original_prime
            original_element.sub = original_sub
        else:
            original_element = None
        return original_element, copied_element

    def _copy_and_modify_node_for_split(self, original_node, variable, current_depth, max_depth):
        if original_node.num_parents == 0:
            raise ValueError("Some node does not have a parent.")
        original_node.decrease_num_parents_by_one()
        if isinstance(original_node, CircuitTerminal):
            if original_node.var_index == variable:
                if original_node.var_value == LITERAL_ALWAYS_SATISFIED:
                    original_node = self._terminal_nodes[variable - 1]
                    copied_node = self._terminal_nodes[self._num_variables + variable - 1]
                elif original_node.var_value == LITERAL_IS_TRUE:
                    copied_node = None
                elif original_node.var_value == LITERAL_IS_FALSE:
                    original_node = None
                    copied_node = self._terminal_nodes[self._num_variables + variable - 1]
                else:
                    raise ValueError('Under the current setting,'
                                     ' terminal nodes that are unsatisfiable should not exist.')
            else:
                copied_node = original_node
            return original_node, copied_node
        else:
            if original_node.num_parents > 0:
                original_node = self._deep_copy_node(original_node, variable, current_depth, max_depth)
            copied_elements = []
            i = 0
            while i < len(original_node.elements):
                original_element, copied_element = \
                    self._copy_and_modify_element_for_split(original_node.elements[i],
                                                            variable, current_depth + 1, max_depth)
                if original_element is None:
                    original_node.remove_element(i)
                else:
                    i += 1
                if copied_element is not None:
                    copied_elements.append(copied_element)
            if len(copied_elements) == 0:
                copied_node = None
            else:
                self._num_created_nodes += 1
                copied_node = OrGate(original_node.vtree, self._num_created_nodes, copied_elements)
            if len(original_node.elements) == 0:
                original_node = None
            return original_node, copied_node

    def _deep_copy_node(self, node, variable, current_depth, max_depth):
        if isinstance(node, CircuitTerminal):
            return node
        else:
            if len(node.elements) == 0:
                raise ValueError("Decision nodes should have at least one elements.")
            copied_elements = []
            for element in node.elements:
                copied_elements.append(self._deep_copy_element(element, variable, current_depth + 1, max_depth))
            self._num_created_nodes += 1
            return OrGate(node.vtree, self._num_created_nodes, copied_elements)

    def _deep_copy_element(self, element, variable, current_depth, max_depth):
        if current_depth >= max_depth:
            if variable in element.prime.vtree.variables:
                copied_element = AndGate(self._deep_copy_node(element.prime, variable, current_depth, max_depth),
                                         element.sub)
            elif variable in element.sub.vtree.variables:
                copied_element = AndGate(element.prime,
                                         self._deep_copy_node(
                                                element.sub, variable, current_depth, max_depth
                                            ))
            else:
                copied_element = AndGate(element.prime, element.sub)
        else:
            copied_element = AndGate(self._deep_copy_node(element.prime, variable, current_depth, max_depth),
                                     self._deep_copy_node(element.sub, variable, current_depth, max_depth))
        copied_element.splittable_variables = copy.deepcopy(element.splittable_variables)
        copied_element.parameter = element.parameter
        return copied_element

    def calculate_accuracy_precision_recall_and_f1(self, data):
        """Calculate accuracy, precision and recall given the learned parameters on the provided data."""
        y = self.predict(data.positive_image_features)
        true_positive = np.count_nonzero(y > 0.5)
        false_negative = data.positive_images.shape[0] - true_positive

        y = self.predict(data.negative_image_features)
        true_negative = np.count_nonzero(y < 0.5)
        false_positive = data.negative_images.shape[0] - true_negative

        accuracy = (true_positive + true_negative) / (true_positive + false_positive + true_negative + false_negative)
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        f1 = 2 * true_positive / (2 * true_positive + false_positive + false_negative)

        return accuracy, precision, recall, f1

    def predict(self, features):
        """Predict the given images."""
        y = 1.0 / (1.0 + np.exp(-(features * self._parameters).sum(axis=1)))
        return y


    def learn_parameters(self, data, num_iterations):
        """Logistic Psdd's parameter learning is reduced to logistic regression.
        We use mini-batch SGD to optimize the parameters."""
        model = LogisticRegression(solver='saga', fit_intercept=False, max_iter=num_iterations,
                                   C=1e8,  warm_start=True, coef_=[self._parameters])
        images, features, labels = data.balanced_all()
        model.fit(features, labels.ravel())
        self._record_learned_parameters(model.coef_[0])

    def change_structure(self, data, depth, num_splits):
        images, features, labels = data.balanced_all()
        splits = self._select_element_and_variable_to_split(images, features,
                                                            labels, num_splits)
        for element_to_split, variable_to_split in splits:
            if not element_to_split.flag:
                self._split(element_to_split, variable_to_split, depth)
        self._serialize()
