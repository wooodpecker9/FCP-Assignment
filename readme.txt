Repository link: https://github.com/wooodpecker9/FCP-Assignment.git
Requirements:
- Install matplotlib
- Install numpy

Usage:
Run FCP-assignment.py along with one of the following arguments:

Arguments:
- -ising_model: Run the Ising Model simulation.
- -alpha <alpha>: Set the temperature parameter for the Ising Model (default: 1.0).
- -external <H>: Set the external pull value for the Ising Model (default: 0.0).
- -test_ising: Run tests for the Ising Model.
- -defuant: Run the Defuant Model simulation.
- -beta <beta>: Set the beta parameter for the Defuant Model (default: 0.2).
- -threshold <threshold>: Set the threshold parameter for the Defuant Model (default: 0.2).
- -ring_network <N>: Create a ring network of size N.
- -small_world <N>: Create a small-world network of size N.
- -re_wire <probability>: Set the rewiring probability for the small-world network (default: 0.2).
- -use_network <N>: Simulate opinion propagation within a small-world network of size N (only works after -defuant).

Note: Replace <alpha>, <H>, <beta>, <threshold>, and <N> with appropriate numerical values.
