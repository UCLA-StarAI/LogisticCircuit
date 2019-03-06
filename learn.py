from algo.LogisticCircuit import LogisticCircuit
from structure.Vtree import Vtree
from util.mnist_data import read_data_sets
import argparse

FLAGS = None


def main():
    # read dataset and vtree
    data = read_data_sets(FLAGS.data_path, FLAGS.positive_label, FLAGS.percentage)
    vtree = Vtree.read(FLAGS.vtree)

    # create a logistic circuit
    circuit = LogisticCircuit(vtree)

    best_accuracy = float('-inf')
    for i in range(FLAGS.num_structure_learning_iterations):
        data.train.positive_image_features = circuit.calculate_features(data.train.positive_images)
        data.train.negative_image_features = circuit.calculate_features(data.train.negative_images)
        data.test.positive_image_features = circuit.calculate_features(data.test.positive_images)
        data.test.negative_image_features = circuit.calculate_features(data.test.negative_images)
        if i % 10 == 0:
            circuit.learn_parameters(data.train, 3 * FLAGS.num_parameter_learning_iterations)
        else:
            circuit.learn_parameters(data.train, FLAGS.num_parameter_learning_iterations)

        with open(FLAGS.log_file, 'a+') as log_file:
            accuracy, precision, recall, f1 = circuit.calculate_accuracy_precision_recall_and_f1(data.train)
            print("Training accuracy: %f\tprecision: %f\trecall: %f\tf1: %f" %
                  (accuracy, precision, recall, f1))
            log_file.write("Training accuracy: %f\tprecision: %f\trecall: %f\tf1: %f\n" %
                           (accuracy, precision, recall, f1))
            accuracy, precision, recall, f1 = circuit.calculate_accuracy_precision_recall_and_f1(data.test)
            print("Testing accuracy: %f\tprecision: %f\trecall: %f\tf1: %f" %
                  (accuracy, precision, recall, f1))
            log_file.write("Testing accuracy: %f\tprecision: %f\trecall: %f\tf1: %f\n" %
                           (accuracy, precision, recall, f1))
            print("Num parameters: %d" % circuit.num_parameters)
            log_file.write("Num parameters: %d\n" % circuit.num_parameters)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            with open(FLAGS.prediction_file, 'w') as prediction_file:
                test_image_feature = circuit.calculate_features(data.test.images)
                predictions = circuit.predict(test_image_feature)
                prediction_file.write('\n'.join(str(p) for p in predictions))

        circuit.change_structure(data.train, FLAGS.depth, FLAGS.num_splits)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str,
                        help='Directory for storing input data')
    parser.add_argument('--vtree', type=str,
                        default='balanced.vtree',
                        help='Path for vtree')
    parser.add_argument('--positive_label', type=int)
    parser.add_argument('--num_structure_learning_iterations', type=int,
                        default=5000)
    parser.add_argument('--num_parameter_learning_iterations', type=int,
                        default=15)
    parser.add_argument('--depth', type=int,
                        default=3)
    parser.add_argument('--num_splits', type=int,
                        default=3)
    parser.add_argument('--log_file', type=str)
    parser.add_argument('--prediction_file', type=str)
    parser.add_argument('--percentage', type=float,
                        default=1.0,
                        help='Percentage of the training dataset to be used.')
    FLAGS = parser.parse_args()
    main()
