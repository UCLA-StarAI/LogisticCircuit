'''
Generate a balanced vtree.
It takes two arguments: 1) the number of input variables in your dataset 2) the file name to store the vtree.
'''

import argparse


VTREE_FORMAT = """c ids of vtree nodes start at 0
c ids of variables start at 1
c vtree nodes appear bottom-up, children before parents
c
c file syntax:
c vtree number-of-nodes-in-vtree
c L id-of-leaf-vtree-node id-of-variable
c I id-of-internal-vtree-node id-of-left-child id-of-right-child
c
"""


FLAGS = None


def main():
    with open(FLAGS.vtree_file, 'w') as f_out:
        f_out.write(VTREE_FORMAT)

        num_nodes = FLAGS.num_variables
        num_to_be_paired_nodes = FLAGS.num_variables
        while num_to_be_paired_nodes > 1:
            num_nodes += num_to_be_paired_nodes // 2
            num_to_be_paired_nodes -= num_to_be_paired_nodes // 2
        f_out.write(f'vtree {num_nodes}\n')

        to_be_paired_nodes = []
        for i in range(FLAGS.num_variables):
            f_out.write(f'L {i} {i+1}\n')
            to_be_paired_nodes.append(i)
        index = FLAGS.num_variables
        while len(to_be_paired_nodes) > 1:
            f_out.write(f'I {index} {to_be_paired_nodes[0]} {to_be_paired_nodes[1]}\n')
            to_be_paired_nodes = to_be_paired_nodes[2:]
            to_be_paired_nodes.append(index)
            index += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_variables', type=int,
                        help='The number of input variables in your dataset')
    parser.add_argument('--vtree_file', type=str,
                        help='The file name to store the generated vtee')
    FLAGS = parser.parse_args()
    main()
