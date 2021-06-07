from cuda import mmult
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import math
from copy import *


### NOTE: The only methods you are required to have are:
#   * predict
#   * fit
#   * score

# noinspection PyMethodMayBeStatic
class TreeNode:
    def __init__(self, prev, prev_node_decision, column_number, output_class, next_nodes):
        self.decision = prev_node_decision
        self.column_number = column_number
        self.output_class = output_class
        self.next_nodes = next_nodes


class ID3Classifier(BaseEstimator, ClassifierMixin):

    def __init__(self):
        """ Initialize class with chosen hyperparameters.
        Args:
        """
        self.tree = TreeNode(None, None, None, None, None)
        self.X = None
        self.X_types = None
        self.y = None
        self.y_types = None
        self.arrays = []
        self.free_arrays = []

    def fit(self, X, y, column_types=None, target_types=None):
        """ Fit the data; Make the Decision tree

        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
            y (array-like): A 2D numpy array with the training targets

        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)

        """
        self.tree = TreeNode(None, None, None, None, None)
        self.X = X
        self.y = y
        self.X_types = self.get_unique_values(X)
        self.y_types = self.get_unique_values(y)
        self._make_nodes(self.tree, [], [])
        return self

    def _make_nodes(self, current_node, columns_used, rows_removed):
        target = np.delete(self.y, rows_removed, 0)
        target_types = self.get_unique_values(target)[0]
        if len(target_types) == 1:
            current_node.next_nodes = None
            current_node.output_class = target_types[0]
            return
        elif len(set(rows_removed)) == self.X.shape[0]:
            current_node.next_nodes = None
            current_node.output_class = self.y_types[0]
            return
        elif len(columns_used) == self.X.shape[1]:
            current_node.next_nodes = None
            converted_target = np.resize(np.array(target, np.dtype('i8')), (target.shape[0]))
            current_node.output_class = np.argmax(np.bincount(converted_target))
            return
        columns = np.delete(self.X, columns_used, 1)
        columns = np.delete(columns, rows_removed, 0)
        column_types = copy(self.X_types)
        for i in range(len(column_types) - 1, -1, -1):
            if i in columns_used:
                column_types.pop(i)

        best_column = self._get_best_column(columns, column_types, target, target_types)
        for i in columns_used:
            if best_column >= i:
                best_column += 1

        columns_used.append(best_column)
        columns_used.sort()
        current_node.column_number = best_column
        current_node.next_nodes = []
        for i in self.X_types[best_column]:
            new_node = TreeNode(current_node, i, None, None, None)
            current_node.next_nodes.append(new_node)
            new_rows_removed = np.nonzero(self.X[:, best_column] != i)[0]
            rows_removed.extend(new_rows_removed)
            self._make_nodes(new_node, columns_used, rows_removed)
            for row in new_rows_removed:
                rows_removed.remove(row)

        columns_used.remove(best_column)

    def _get_best_column(self, columns, column_types, target, target_types):
        """

        Args:
            columns:
            column_types:
            target:
            target_types:

        Returns:

        """
        info_gain = []
        for i in range(0, columns.shape[1] + 1):
            # This gets base info gain
            if i == columns.shape[1]:
                column = np.ones(columns.shape[0])
                curr_column_types = [1]
            else:
                column = columns[:, i]
                curr_column_types = column_types[i]
            info_gain.append(self._calc_info_gain(column, curr_column_types, target, target_types))
        last_element = len(info_gain) - 1
        for i in range(0, last_element):
            info_gain[i] = info_gain[last_element] - info_gain[i]
        info_gain.pop(last_element)

        best_index = 0
        for i in range(0, len(info_gain)):
            if info_gain[i] > info_gain[best_index]:
                best_index = i
        return best_index

    def get_unique_values(self, columns):
        """

        Args:
            columns:

        Returns:
            a list with each index containing an array of unique values in that column

        """
        if columns.ndim == 1:
            columns = np.resize(columns, (columns.shape[0], 1))

        unique_values = []
        for i in range(0, columns.shape[1]):
            unique_values.append(np.unique(columns[:, i]))
        return unique_values

    def _calc_info_gain(self, features, feature_type, targets, target_type):
        """

        Args:
            features (numpy array (r,)): 1D array with every row and only has column with the feature we are testing
            targets (numpy array(r,)): 1D array with the targets
            feature_type: Assuming that the lowest type number is 0
            target_type: Assuming that the lowest type number is 0
        Returns:

        """
        num_rows = features.shape[0]
        features = np.resize(features, (num_rows, 1))
        targets = np.resize(targets, (num_rows, 1))
        target_type = np.resize(np.asarray(target_type), (1, len(target_type)))
        feature_type = np.resize(np.asarray(feature_type), (1, len(feature_type)))

        targets_by_type, targets_by_type_id = self._fill_type_matrix(targets, target_type)
        features_by_type, features_by_type_id = self._fill_type_matrix(features, feature_type)

        count_matrix_id = self.allocate_array((targets_by_type.shape[1], features_by_type.shape[1]))
        count_matrix = self.arrays[count_matrix_id]
        transposed_targets_by_type_id = self.allocate_array((targets_by_type.shape[1], targets_by_type.shape[0]))
        transposed_targets_by_type = self.arrays[transposed_targets_by_type_id]
        transposed_targets_by_type[:, :] = targets_by_type.transpose()
        mmult.mmult(transposed_targets_by_type, features_by_type, count_matrix)  # matrix shows how many of a feature type has a target type, (target, feature)
        self.free_array(targets_by_type_id)
        self.free_array(features_by_type_id)
        self.free_array(transposed_targets_by_type_id)
        num_features = np.sum(count_matrix, axis=0, keepdims=True)
        features_per_count_id = self.allocate_array((target_type.shape[1], feature_type.shape[1]))
        features_per_count = self.arrays[features_per_count_id]
        features_per_count[:, :] = num_features
        ones_id = self.allocate_array((len(target_type), len(feature_type)))
        ones = self.arrays[ones_id]
        ones[:, :] = 1
        # For times where there isn't any output represented in the features, I'm not using np.where because it doesn't operate in place
        np.add(features_per_count, ones, out=features_per_count, where=features_per_count == 0)

        np.divide(count_matrix, features_per_count, out=features_per_count)
        # changing all zeros to ones to prevent domain error, log(1) == 0 so the result will not change
        np.add(features_per_count, ones, out=features_per_count, where=features_per_count == 0)

        np.log2(features_per_count, out=count_matrix)
        np.multiply(features_per_count, count_matrix, out=features_per_count)
        np.divide(num_features, num_rows * -1, out=num_features)
        info_gain = np.sum(features_per_count, axis=0, keepdims=True)
        np.multiply(info_gain, num_features, out=info_gain)
        self.free_array(count_matrix_id)
        self.free_array(features_per_count_id)
        self.free_array(ones_id)
        return np.sum(info_gain)

    def _fill_type_matrix(self, data, types):
        num_rows = data.shape[0]
        num_types = types.shape[1]
        dst_id = self.allocate_array((num_rows, num_types))
        dst = self.arrays[dst_id]
        expanded_types_id = self.allocate_array((num_rows, num_types))
        expanded_types = self.arrays[expanded_types_id]
        expanded_data_id = self.allocate_array((num_rows, num_types))
        expanded_data = self.arrays[expanded_data_id]
        ones_id = self.allocate_array((num_rows, num_types))
        ones = self.arrays[ones_id]
        dst[:, :] = 0
        expanded_types[:, :] = types
        expanded_data[:, :] = data
        ones[:, :] = 1
        np.add(dst, ones, out=dst, where=expanded_data == expanded_types)
        self.free_array(expanded_data_id)
        self.free_array(expanded_types_id)
        self.free_array(ones_id)
        return dst, dst_id

    def _get_max_array_shape(self):
        max_types = 0
        for types in self.X_types:
            if len(types) > max_types:
                max_types = len(types)
        if len(self.y_types) > max_types:
            max_types = len(self.y_types)
        return self.X.shape[0], max_types

    def predict(self, X):
        """ Predict all classes for a dataset X

        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets

        Returns:
            array, shape (n_samples,)
                Predicted target values per element in X.
        """
        results = np.zeros((X.shape[0]))
        for i in range(0, X.shape[0]):
            current_input = X[i, :]
            current_node = self.tree
            j = 0
            while current_node.next_nodes is not None:
                j += 1
                type_chosen = current_input[current_node.column_number]
                for node in current_node.next_nodes:
                    if node.decision == type_chosen:
                        current_node = node
                        break
            results[i] = current_node.output_class
        return results

    def score(self, X, y):
        """ Return accuracy(Classification Acc) of model on a given dataset. Must implement own score function.

        Args:
            X (array-like): A 2D numpy array with data, excluding targets
            y (array-like): A 2D numpy array of the targets 
        """
        results = self.predict(X)
        y = np.resize(y, (y.shape[0]))
        results -= y
        results = results == 0
        return np.sum(results) / float(results.shape[0])

    def allocate_array(self, shape):
        for i in range(0, len(self.free_arrays)):
            if self.free_arrays[i] == 1 and self.arrays[i].shape == shape:
                self.free_arrays[i] = 0
                return i
        self.arrays.append(np.ones(shape, dtype=np.float32))
        self.free_arrays.append(0)
        return len(self.free_arrays) - 1

    def free_array(self, id):
        self.free_arrays[id] = 1
