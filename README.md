# LogisticCircuit

This repo contains the code to run experiments reported in the paper "Learning Logistic Circuits", published in AAAI 2019.

We kindly include a help function in learn.py. In other words, to query what arguments are required and the detailed description what each argument is for, please execute "python3 learn.py --help" in your terminal.

Note the default balanced.vtree is for MNIST and Fashion-MNIST. To run experiments on other datasets, a different vtree is necessary. As requested by some users, we include a small script (generate_balanced_vtree.py) in "util/" to generate balanced vtrees. The generated vtrees from this script are not optimized, and thus do not guarantee optimal performance.

A description of parameter-tuning details to exactly reproduce the paper experiments to follow shortly.

For questions, please don't hesitate to send us an email at yliang@cs.ucla.edu
