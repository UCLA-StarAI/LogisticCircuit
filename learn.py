import argparse
import time
import warnings

from algo.LogisticCircuit import LogisticCircuit
from structure.Vtree import Vtree
from util.mnist_data import read_data_sets

FLAGS = None


def main():
    # read dataset and vtree
    data = read_data_sets(FLAGS.data_path, FLAGS.percentage)
    vtree = Vtree.read(FLAGS.vtree)

    # create a logistic circuit
    if FLAGS.circuit != "":
        with open(FLAGS.circuit, "r") as circuit_file:
            circuit = LogisticCircuit(vtree, FLAGS.num_classes, circuit_file=circuit_file)
            print("The saved circuit is successfully loaded.")
            data.train.features = circuit.calculate_features(data.train.images)
    else:
        circuit = LogisticCircuit(vtree, FLAGS.num_classes)
        data.train.features = circuit.calculate_features(data.train.images)
        circuit.learn_parameters(data.train, 50)

    print(f"The starting circuit has {circuit.num_parameters} parameters.")
    data.valid.features = circuit.calculate_features(data.valid.images)
    data.test.features = circuit.calculate_features(data.test.images)
    valid_accuracy = circuit.calculate_accuracy(data.valid)
    print(
        f"Its performance is as follows. "
        f"Training accuracy: {circuit.calculate_accuracy(data.train):.5f}\t"
        f"Valid accuracy: {valid_accuracy:.5f}\t"
        f"Test accuracy: {circuit.calculate_accuracy(data.test):.5f}"
    )

    print("Start structure learning.")

    best_accuracy = valid_accuracy
    for i in range(FLAGS.num_structure_learning_iterations):
        cur_time = time.time()

        circuit.change_structure(data.train, FLAGS.depth, FLAGS.num_splits)

        data.train.features = circuit.calculate_features(data.train.images)
        data.valid.features = circuit.calculate_features(data.valid.images)
        data.test.features = circuit.calculate_features(data.test.images)

        circuit.learn_parameters(data.train, FLAGS.num_parameter_learning_iterations)

        valid_accuracy = circuit.calculate_accuracy(data.valid)
        print(
            f"Training accuracy: {circuit.calculate_accuracy(data.train):.5f}\t"
            f"Valid accuracy: {valid_accuracy:.5f}\t"
            f"Test accuracy: {circuit.calculate_accuracy(data.test):.5f}"
        )
        print(f"Num parameters: {circuit.num_parameters}\tTime spent: {(time.time() - cur_time):.2f}")

        if FLAGS.save_path != "" and (valid_accuracy > best_accuracy):
            best_accuracy = valid_accuracy
            print("Obtained a logistic circuit with higher classification accuracy. Start saving.")
            with open(FLAGS.save_path, "w") as circuit_file:
                circuit.save(circuit_file)
            print("Logistic circuit saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="Directory for the stored input data.")
    parser.add_argument("--num_classes", type=int, help="Number of classes in the classification task.")
    parser.add_argument("--vtree", type=str, default="balanced.vtree", help="Path for vtree.")
    parser.add_argument(
        "--circuit",
        type=str,
        default="",
        help="[Optional] File path for the saved logistic circuit to load. "
        "Note this circuit has to be based on the same vtree as provided in --vtree.",
    )
    parser.add_argument(
        "--num_structure_learning_iterations",
        type=int,
        default=5000,
        help="[Optional] Num of iterations for structure learning. Its default value is 5000.",
    )
    parser.add_argument(
        "--num_parameter_learning_iterations",
        type=int,
        default=15,
        help="[Optional] Number of iterations for parameter learning after the structure is changed."
        "Its default value is 15.",
    )
    parser.add_argument("--depth", type=int, default=2, help="[Optional] The depth of every split. Its default value is 2.")
    parser.add_argument(
        "--num_splits",
        type=int,
        default=3,
        help="[Optional] The number of splits in one iteration of structure learning." "It default value is 3.",
    )
    parser.add_argument(
        "--percentage",
        type=float,
        default=1.0,
        help="[Optional] The percentage of the training dataset that will be used. " "Its default value is 100%%.",
    )
    parser.add_argument("--save_path", type=str, default="", help="[Optional] File path to save the best-performing circuit.")
    FLAGS = parser.parse_args()
    if FLAGS.num_classes == 2:
        FLAGS.num_classes = 1
        message = (
            "It is essentially a binary classification task when num_classes is set to 2, "
            + "and hence we automatically modify it to be 1 to be better compatible with sklearn."
        )
        warnings.warn(message, stacklevel=2)
    main()
