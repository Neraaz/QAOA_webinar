import cudaq
from cudaq import spin
import numpy as np

# The two graph inputs from the paper

# BIG 1

nodes = [0, 1, 2, 3, 4, 5]
qubit_num = len(nodes)
edges = [[0, 1], [0, 2], [0, 4], [0, 5], [1, 2], [1, 3], [1, 5], [2, 3], [2, 4],
         [3, 4], [3, 5], [4, 5]]
non_edges = [
    [u, v] for u in nodes for v in nodes if u < v and [u, v] not in edges
]

print('Edges: ', edges)
print('Non-Edges: ', non_edges)

weights = [0.6686, 0.6686, 0.6686, 0.1453, 0.1453, 0.1453]
penalty = 6.0
num_layers = 3

# BIG 2 (More expensive simulation)
#nodes=[0,1,2,3,4,5,6,7]
#qubit_num=len(nodes)
#edges=[[0,1],[0,2],[0,5],[0,6],[0,7],[1,2],[1,4],[1,6],[1,7],[2,4],[2,5],[2,7],[3,4],[3,5],[3,6],\
#    [4,5],[4,6],[5,6]]
#non_edges=[[u,v] for u in nodes for v in nodes if u<v and [u,v] not in edges]
#print('Edges: ', edges)
#print('Non-edges: ', non_edges)
#weights=[0.6686,0.6686,0.6886,0.1091,0.0770,0.0770,0.0770,0.0770]
#penalty=8.0
#num_layers=8

# Generate the Hamiltonian
cudaq.set_target('nvidia')
def ham_clique(penalty, nodes, weights, non_edges) -> cudaq.SpinOperator:

    spin_ham = 0
    for wt, node in zip(weights, nodes):
        #print(wt,node)
        spin_ham += 0.5 * wt * spin.z(node)
        spin_ham -= 0.5 * wt * spin.i(node)

    for non_edge in non_edges:
        u, v = (non_edge[0], non_edge[1])
        #print(u,v)
        spin_ham += penalty / 4.0 * (spin.z(u) * spin.z(v) - spin.z(u) -
                                     spin.z(v) + spin.i(u) * spin.i(v))

    return spin_ham

# Collect coefficients from a spin operator so we can pass them to a kernel
def term_coefficients(ham: cudaq.SpinOperator) -> list[complex]:
    result = []
    for term in ham:
        result.append(term.evaluate_coefficient())
    return result

    # Collect Pauli words from a spin operator so we can pass them to a kernel


def term_words(ham: cudaq.SpinOperator) -> list[str]:
    # Our kernel uses these words to apply exp_pauli to the entire state.
    # we hence ensure that each pauli word covers the entire space.

    result = []
    for term in ham:
        result.append(term.get_pauli_word(qubit_num))
    return result


ham = ham_clique(penalty, nodes, weights, non_edges)
print(ham)

coef = term_coefficients(ham)
words = term_words(ham)

print(term_coefficients(ham))
print(term_words(ham))

@cudaq.kernel
def dc_qaoa(qubit_num:int, num_layers:int, thetas:list[float],\
    coef:list[complex], words:list[cudaq.pauli_word]):

    qubits = cudaq.qvector(qubit_num)

    h(qubits)

    count = 0
    for p in range(num_layers):

        for i in range(len(coef)):
            exp_pauli(thetas[count] * coef[i].real, qubits, words[i])
            count += 1

        for j in range(qubit_num):
            rx(thetas[count], qubits[j])
            count += 1

        #Comment out this for loop for conventional QAOA
        for k in range(qubit_num):
            ry(thetas[count], qubits[k])
            count += 1

# Specify the optimizer and its initial parameters.
optimizer = cudaq.optimizers.NelderMead()

#Specify random seeds
np.random.seed(13)
cudaq.set_random_seed(13)

# if dc_qaoa used
parameter_count = (2 * qubit_num + len(coef)) * num_layers

# if qaoa used
# parameter_count=(qubit_num+len(coef))*num_layers

print('Total number of parameters: ', parameter_count)
optimizer.initial_parameters = np.random.uniform(-np.pi / 8, np.pi / 8,
                                                 parameter_count)
print("Initial parameters = ", optimizer.initial_parameters)

cost_values = []


def objective(parameters):

    cost = cudaq.observe(dc_qaoa, ham, qubit_num, num_layers, parameters, coef,
                         words).expectation()
    cost_values.append(cost)
    return cost


# Optimize!
optimal_expectation, optimal_parameters = optimizer.optimize(
    dimensions=parameter_count, function=objective)

print('optimal_expectation =', optimal_expectation)
print('optimal_parameters =', optimal_parameters)

shots = 200000

counts = cudaq.sample(dc_qaoa,
                      qubit_num,
                      num_layers,
                      optimal_parameters,
                      coef,
                      words,
                      shots_count=shots)
print(counts)

print('The MVWCP is given by the partition: ', counts.most_probable())

import matplotlib.pyplot as plt

x_values = list(range(len(cost_values)))
y_values = cost_values

plt.plot(x_values, y_values)

plt.xlabel("Epochs")
plt.ylabel("Cost Value")
plt.savefig("x.png")
