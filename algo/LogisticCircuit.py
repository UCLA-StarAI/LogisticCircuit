import copy
import gc
from collections import deque

import numpy as np

from algo.LogisticRegression import LogisticRegression
from structure.AndGate import AndGate
from structure.CircuitNode import CircuitNode, OrGate, CircuitTerminal
from structure.CircuitNode import LITERAL_IS_TRUE, LITERAL_IS_FALSE
from structure.Vtree import Vtree

FORMAT = """c variables (from inputs) start from 1
c ids of logistic circuit nodes start from 0
c nodes appear bottom-up, children before parents
c the last line of the file records the bias parameter
c three types of nodes:
c	T (terminal nodes that correspond to true literals)
c	F (terminal nodes that correspond to false literals)
c	D (OR gates)
c
c file syntax:
c Logisitic Circuit
c T id-of-true-literal-node id-of-vtree variable parameters
c F id-of-false-literal-node id-of-vtree variable parameters
c D id-of-or-gate id-of-vtree number-of-elements (id-of-prime id-of-sub parameters)s
c B bias-parameters
c
"""


class LogisticCircuit(object):
    def __init__(self, vtree, num_classes, circuit_file=None):
        self._vtree = vtree
        self._num_classes = num_classes
        self._largest_index = -1
        self._num_variables = vtree.var_count

        self._terminal_nodes = [None] * 2 * self._num_variables
        self._decision_nodes = None
        self._elements = None
        self._parameters = None
        self._bias = np.random.random_sample(size=(num_classes,))

        if circuit_file is None:
            self._generate_all_terminal_nodes(vtree)
            self._root = self._new_logistic_psdd(vtree)
        else:
            self._root = self.load(circuit_file)

        self._serialize()

    @property
    def vtree(self):
        return self._vtree

    @property
    def num_parameters(self):
        return self._parameters.size

    @property
    def parameters(self):
        return self._parameters

    def _generate_all_terminal_nodes(self, vtree: Vtree):
        if vtree.is_leaf():
            var_index = vtree.var
            self._largest_index += 1
            self._terminal_nodes[var_index - 1] = CircuitTerminal(
                self._largest_index, vtree, var_index, LITERAL_IS_TRUE, np.random.random_sample(size=(self._num_classes,))
            )
            self._largest_index += 1
            self._terminal_nodes[self._num_variables + var_index - 1] = CircuitTerminal(
                self._largest_index, vtree, var_index, LITERAL_IS_FALSE, np.random.random_sample(size=(self._num_classes,))
            )
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
            elements.append(
                AndGate(
                    self._terminal_nodes[prime_variable - 1],
                    self._terminal_nodes[sub_variable - 1],
                    np.random.random_sample(size=(self._num_classes,)),
                )
            )
            elements.append(
                AndGate(
                    self._terminal_nodes[prime_variable - 1],
                    self._terminal_nodes[self._num_variables + sub_variable - 1],
                    np.random.random_sample(size=(self._num_classes,)),
                )
            )
            elements.append(
                AndGate(
                    self._terminal_nodes[self._num_variables + prime_variable - 1],
                    self._terminal_nodes[sub_variable - 1],
                    np.random.random_sample(size=(self._num_classes,)),
                )
            )
            elements.append(
                AndGate(
                    self._terminal_nodes[self._num_variables + prime_variable - 1],
                    self._terminal_nodes[self._num_variables + sub_variable - 1],
                    np.random.random_sample(size=(self._num_classes,)),
                )
            )
        elif left_vtree.is_leaf():
            elements.append(
                AndGate(
                    self._terminal_nodes[prime_variable - 1],
                    self._new_logistic_psdd(right_vtree),
                    np.random.random_sample(size=(self._num_classes,)),
                )
            )
            elements.append(
                AndGate(
                    self._terminal_nodes[self._num_variables + prime_variable - 1],
                    self._new_logistic_psdd(right_vtree),
                    np.random.random_sample(size=(self._num_classes,)),
                )
            )
            for element in elements:
                element.splittable_variables = copy.deepcopy(right_vtree.variables)
        elif right_vtree.is_leaf():
            elements.append(
                AndGate(
                    self._new_logistic_psdd(left_vtree),
                    self._terminal_nodes[sub_variable - 1],
                    np.random.random_sample(size=(self._num_classes,)),
                )
            )
            elements.append(
                AndGate(
                    self._new_logistic_psdd(left_vtree),
                    self._terminal_nodes[self._num_variables + sub_variable - 1],
                    np.random.random_sample(size=(self._num_classes,)),
                )
            )
            for element in elements:
                element.splittable_variables = copy.deepcopy(left_vtree.variables)
        else:
            elements.append(
                AndGate(
                    self._new_logistic_psdd(left_vtree),
                    self._new_logistic_psdd(right_vtree),
                    np.random.random_sample(size=(self._num_classes,)),
                )
            )
            elements[0].splittable_variables = copy.deepcopy(vtree.variables)
        self._largest_index += 1
        root = OrGate(self._largest_index, vtree, elements)
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
        self._parameters = self._bias.reshape(-1, 1)
        for terminal_node in self._terminal_nodes:
            self._parameters = np.concatenate((self._parameters, terminal_node.parameter.reshape(-1, 1)), axis=1)
        for element in self._elements:
            self._parameters = np.concatenate((self._parameters, element.parameter.reshape(-1, 1)), axis=1)
        gc.collect()

    def _record_learned_parameters(self, parameters):
        self._parameters = copy.deepcopy(parameters)
        self._bias = self._parameters[:, 0]
        for i in range(len(self._terminal_nodes)):
            self._terminal_nodes[i].parameter = self._parameters[:, i + 1]
        for i in range(len(self._elements)):
            self._elements[i].parameter = self._parameters[:, i + 1 + 2 * self._num_variables]
        gc.collect()

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

    def _select_element_and_variable_to_split(self, data, num_splits):
        y = self.predict_prob(data.features)
        if self._num_classes == 1:
            y = np.hstack(((1.0 - y).reshape(-1, 1), y.reshape(-1, 1)))

        delta = data.one_hot_labels - y
        element_gradients = np.stack(
            [
                (delta[:, i].reshape(-1, 1) * data.features)[:, 2 * self._num_variables + 1 :]
                for i in range(self._num_classes)
            ],
            axis=0,
        )
        element_gradient_variance = np.var(element_gradients, axis=1)
        element_gradient_variance = np.average(element_gradient_variance, axis=0)

        candidates = sorted(
            zip(self._elements, element_gradient_variance, data.features.T[2 * self._num_variables + 1 :]),
            reverse=True,
            key=lambda x: x[1],
        )
        selected = []
        for candidate in candidates[: min(5000, len(candidates))]:
            element_to_split = candidate[0]
            if len(element_to_split.splittable_variables) > 0 and np.sum(candidate[2]) > 25:
                original_feature = candidate[2]
                original_variance = candidate[1]
                variable_to_split = None
                min_after_split_variance = float("inf")
                for variable in element_to_split.splittable_variables:
                    left_feature = original_feature * data.images[:, variable - 1]
                    right_feature = original_feature - left_feature

                    if np.sum(left_feature) > 10 and np.sum(right_feature) > 10:

                        left_gradient = (data.one_hot_labels - y) * left_feature.reshape((-1, 1))
                        right_gradient = (data.one_hot_labels - y) * right_feature.reshape((-1, 1))

                        w = np.sum(data.images[:, variable - 1]) / data.num_samples

                        after_split_variance = w * np.average(np.var(left_gradient, axis=0)) + (1 - w) * np.average(
                            np.var(right_gradient, axis=0)
                        )
                        if after_split_variance < min_after_split_variance:
                            min_after_split_variance = after_split_variance
                            variable_to_split = variable
                if min_after_split_variance < original_variance:
                    improved_amount = min_after_split_variance - original_variance
                    if len(selected) == num_splits:
                        if improved_amount < selected[0][1]:
                            selected = selected[1:]
                            selected.append(((element_to_split, variable_to_split), improved_amount))
                            selected.sort(key=lambda x: x[1])
                    else:
                        selected.append(((element_to_split, variable_to_split), improved_amount))
                        selected.sort(key=lambda x: x[1])

        gc.collect()
        return [x[0] for x in selected]

    def _split(self, element_to_split, variable_to_split, depth):
        parent = element_to_split.parent
        original_element, copied_element = self._copy_and_modify_element_for_split(
            element_to_split, variable_to_split, 0, depth
        )
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
                original_prime, copied_prime = self._copy_and_modify_node_for_split(
                    original_prime, variable, current_depth, max_depth
                )
                copied_sub = original_sub
            elif variable in original_sub.vtree.variables:
                original_sub, copied_sub = self._copy_and_modify_node_for_split(
                    original_sub, variable, current_depth, max_depth
                )
                copied_prime = original_prime
            else:
                copied_prime = original_prime
                copied_sub = original_sub
        else:
            original_prime, copied_prime = self._copy_and_modify_node_for_split(
                original_prime, variable, current_depth, max_depth
            )
            original_sub, copied_sub = self._copy_and_modify_node_for_split(original_sub, variable, current_depth, max_depth)
        if copied_prime is not None and copied_sub is not None:
            copied_element = AndGate(copied_prime, copied_sub, copy.deepcopy(original_element.parameter))
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
                if original_node.var_value == LITERAL_IS_TRUE:
                    copied_node = None
                elif original_node.var_value == LITERAL_IS_FALSE:
                    original_node = None
                    copied_node = self._terminal_nodes[self._num_variables + variable - 1]
                else:
                    raise ValueError(
                        "Under the current setting,"
                        "we only support terminal nodes that are either positive or negative literals."
                    )
            else:
                copied_node = original_node
            return original_node, copied_node
        else:
            if original_node.num_parents > 0:
                original_node = self._deep_copy_node(original_node, variable, current_depth, max_depth)
            copied_elements = []
            i = 0
            while i < len(original_node.elements):
                original_element, copied_element = self._copy_and_modify_element_for_split(
                    original_node.elements[i], variable, current_depth + 1, max_depth
                )
                if original_element is None:
                    original_node.remove_element(i)
                else:
                    i += 1
                if copied_element is not None:
                    copied_elements.append(copied_element)
            if len(copied_elements) == 0:
                copied_node = None
            else:
                self._largest_index += 1
                copied_node = OrGate(self._largest_index, original_node.vtree, copied_elements)
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
            self._largest_index += 1
            return OrGate(self._largest_index, node.vtree, copied_elements)

    def _deep_copy_element(self, element, variable, current_depth, max_depth):
        if current_depth >= max_depth:
            if variable in element.prime.vtree.variables:
                copied_element = AndGate(
                    self._deep_copy_node(element.prime, variable, current_depth, max_depth),
                    element.sub,
                    copy.deepcopy(element.parameter),
                )
            elif variable in element.sub.vtree.variables:
                copied_element = AndGate(
                    element.prime,
                    self._deep_copy_node(element.sub, variable, current_depth, max_depth),
                    copy.deepcopy(element.parameter),
                )
            else:
                copied_element = AndGate(element.prime, element.sub, copy.deepcopy(element.parameter))
        else:
            copied_element = AndGate(
                self._deep_copy_node(element.prime, variable, current_depth, max_depth),
                self._deep_copy_node(element.sub, variable, current_depth, max_depth),
                copy.deepcopy(element.parameter),
            )
        copied_element.splittable_variables = copy.deepcopy(element.splittable_variables)
        return copied_element

    def calculate_accuracy(self, data):
        """Calculate accuracy given the learned parameters on the provided data."""
        y = self.predict(data.features)
        accuracy = np.sum(y == data.labels) / data.num_samples
        return accuracy

    def predict(self, features):
        y = self.predict_prob(features)
        if self._num_classes > 1:
            return np.argmax(y, axis=1)
        else:
            return (y > 0.5).astype(int).ravel()

    def predict_prob(self, features):
        """Predict the given images by providing their corresponding features."""
        y = 1.0 / (1.0 + np.exp(-np.dot(features, self._parameters.T)))
        return y

    def learn_parameters(self, data, num_iterations, num_cores=-1):
        """Logistic Psdd's parameter learning is reduced to logistic regression.
        We use mini-batch SGD to optimize the parameters."""
        model = LogisticRegression(
            solver="saga",
            fit_intercept=False,
            multi_class="ovr",
            max_iter=num_iterations,
            C=0.1,
            warm_start=True,
            tol=1e-5,
            coef_=self._parameters,
            n_jobs=num_cores,
        )
        model.fit(data.features, data.labels)
        self._record_learned_parameters(model.coef_)
        gc.collect()

    def change_structure(self, data, depth, num_splits):
        splits = self._select_element_and_variable_to_split(data, num_splits)
        for element_to_split, variable_to_split in splits:
            if not element_to_split.flag:
                self._split(element_to_split, variable_to_split, depth)
        self._serialize()

    def save(self, f):
        self._serialize()
        f.write(FORMAT)
        f.write(f"Logisitic Circuit\n")
        for terminal_node in self._terminal_nodes:
            terminal_node.save(f)
        for decision_node in reversed(self._decision_nodes):
            decision_node.save(f)
        f.write("B")
        for parameter in self._bias:
            f.write(f" {parameter}")
        f.write("\n")

    def load(self, f):
        # read the format at the beginning
        line = f.readline()
        while line[0] == "c":
            line = f.readline()

        # serialize the vtree
        vtree_nodes = dict()
        unvisited_vtree_nodes = deque()
        unvisited_vtree_nodes.append(self._vtree)
        while len(unvisited_vtree_nodes):
            node = unvisited_vtree_nodes.popleft()
            vtree_nodes[node.index] = node
            if not node.is_leaf():
                unvisited_vtree_nodes.append(node.left)
                unvisited_vtree_nodes.append(node.right)

        # extract the saved logistic circuit
        nodes = dict()
        line = f.readline()
        while line[0] == "T" or line[0] == "F":
            line_as_list = line.strip().split(" ")
            positive_literal, var = (line_as_list[0] == "T"), int(line_as_list[3])
            index, vtree_index = int(line_as_list[1]), int(line_as_list[2])
            parameters = []
            for i in range(self._num_classes):
                parameters.append(float(line_as_list[4 + i]))
            parameters = np.array(parameters, dtype=np.float32)
            if positive_literal:
                nodes[index] = (CircuitTerminal(index, vtree_nodes[vtree_index], var, LITERAL_IS_TRUE, parameters), {var})
            else:
                nodes[index] = (CircuitTerminal(index, vtree_nodes[vtree_index], var, LITERAL_IS_FALSE, parameters), {-var})
            self._largest_index = max(self._largest_index, index)
            line = f.readline()

        self._terminal_nodes = [x[0] for x in nodes.values()]
        self._terminal_nodes.sort(key=lambda x: (-x.var_value, x.var_index))
        if len(self._terminal_nodes) != 2 * self._num_variables:
            raise ValueError(
                "Number of terminal nodes recorded in the circuit file "
                "does not match 2 * number of variables in the provided vtree."
            )

        root = None
        while line[0] == "D":
            line_as_list = line.strip().split(" ")
            index, vtree_index, num_elements = int(line_as_list[1]), int(line_as_list[2]), int(line_as_list[3])
            elements = []
            variables = set()
            for i in range(num_elements):
                prime_index = int(line_as_list[i * (self._num_classes + 2) + 4].strip("("))
                sub_index = int(line_as_list[i * (self._num_classes + 2) + 5])
                element_variables = nodes[prime_index][1].union(nodes[sub_index][1])
                variables = variables.union(element_variables)
                splittable_variables = set()
                for variable in element_variables:
                    if -variable in element_variables:
                        splittable_variables.add(abs(variable))
                parameters = []
                for j in range(self._num_classes):
                    parameters.append(float(line_as_list[i * (self._num_classes + 2) + 6 + j].strip(")")))
                parameters = np.array(parameters, dtype=np.float32)
                elements.append(AndGate(nodes[prime_index][0], nodes[sub_index][0], parameters))
                elements[-1].splittable_variables = splittable_variables
            nodes[index] = (OrGate(index, vtree_nodes[vtree_index], elements), variables)
            root = nodes[index][0]
            self._largest_index = max(self._largest_index, index)
            line = f.readline()

        if line[0] != "B":
            raise ValueError("The last line in a circuit file must record the bias parameters.")
        self._bias = np.array([float(x) for x in line.strip().split(" ")[1:]], dtype=np.float32)

        gc.collect()
        return root
