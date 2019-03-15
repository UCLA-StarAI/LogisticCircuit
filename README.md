# LogisticCircuit

This repo contains the code to run experiments reported in the paper "Learning Logistic Circuits", published in AAAI 2019.

We kindly include a help function in learn.py. In other words, to query what arguments are required and the detailed description what each argument is for, please execute "python3 learn.py --help" in your terminal.

Note the default balanced.vtree is for MNIST and Fashion-MNIST. To run experiments on other datasets, a different vtree is necessary. As requested by some users, we include a small script (generate_balanced_vtree.py) in "util/" to generate balanced vtrees. The generated vtrees from this script are not optimized, and thus do not guarantee optimal performance.

We now also support direct multi-class classification through one single circuit instead of resorting to multiple one-vs-all circuits. We parameterize the same circuit structure with n sets of parameters, each corresponding to one one-vs-all binary classification.

For better reproducibility, we include the trained circuit that achieves the performance reported in our AAAI paper in the folder "pretrained_models". To achieve the optimal classification accuracy, after the structure learning process is finished, we re-learn the parameters with high l2 regularization, and pick the best set of learned parameters according to its result on the validation set. After loading the pretrained circuit, if one keeps running more iterations of parameter learning, he/she may observe a drop of classification accuracy.

For questions, please don't hesitate to send us an email at yliang@cs.ucla.edu
