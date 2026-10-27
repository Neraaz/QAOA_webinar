import numpy as np
import cudaq
from cudaq import spin
from scipy.optimize import minimize
import random

cudaq.set_target("nvidia", option="fp64")
# The Max-cut graph:

# Example 1
#true_energy=-12.0
#nodes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
#edges = [[0,4], [1,4], [0,6], [3,6], [5,6], [2,7], [1,8], [3,8], [5,8], [6,8], [7,8], [2,9], [8,9]]
#weight = [1.0] * len(edges)

# Examle 2:
#true_energy=-5.0
nodes = [0, 1, 2, 3, 4]
edges = [[0, 1], [1, 2], [2, 3], [3, 0], [2, 4], [3, 4]]
weight = [1.0] * len(edges)

# Define the number of qubits.

qubits_num = len(nodes)

# Predefined values to terminate iteration. These can be changed based on user preference
E_prev = 0.0
threshold = 1e-3   # This is a predefined value to stop the calculation. if norm of the gradient smaller than this value, calculation will be terminated.
e_stop = 1e-7      # This is a predefined value to stop the calculation. if dE <= e_stop, calculation stop. dE=current_erg-prev_erg

# Initiate the parameters gamma of the problem Hamiltonian for gradient calculation

init_gamma = 0.01

def spin_ham (edges, weight):

    ham = 0

    count = 0
    for edge in range(len(edges)):

        qubitu = edges[edge][0]
        qubitv = edges[edge][1]
        # Add a term to the Hamiltonian for the edge (u,v)
        ham += 0.5 * (weight[count] * cudaq.spin.z(qubitu) * cudaq.spin.z(qubitv) -
                            weight[count] * cudaq.spin.i(qubitu) * cudaq.spin.i(qubitv))
        count += 1

    return ham

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
        result.append(term.get_pauli_word(qubits_num))
    return result

# Generate the Spin Hamiltonian:

ham = spin_ham(edges, weight)
ham_coef = term_coefficients(ham)
ham_word = term_words(ham)

def qaoa_mixer(n):

    term = spin.x(0)

    for i in range(1, n):
        term = term + spin.x(i)

    pool = [term]
    return pool

def qaoa_single_x(n):

    pool = []

    for i in range(n):
        pool.append(cudaq.SpinOperator(spin.x(i)))

    return pool

def qaoa_double_ops(n):

    pool = []

    for i in range(n-1):
        for j in range(i+1, n):
            pool.append(cudaq.SpinOperator(spin.x(i)) * cudaq.SpinOperator(spin.x(j)))
            pool.append(cudaq.SpinOperator(spin.y(i)) * cudaq.SpinOperator(spin.y(j)))
            pool.append(cudaq.SpinOperator(spin.y(i)) * cudaq.SpinOperator(spin.z(j)))
            pool.append(cudaq.SpinOperator(spin.z(i)) * cudaq.SpinOperator(spin.y(j)))

    return pool

def mixer_pool(qubits_num):
    return (
          qaoa_single_x(qubits_num)
        + qaoa_mixer(qubits_num)
        + qaoa_double_ops(qubits_num)
    )

# Generate the mixer pool

pools = mixer_pool(qubits_num)
print('Number of pool operator: ', len(pools))

# Generate the commutator [H,Ai]

def commutator(pools, ham):
    com_op = []

    for i in range(len(pools)):
    #for i in range(2):
        op = pools[i]
        com_op.append(ham * op - op * ham)

    return com_op

grad_op = commutator(pools, ham)

###########################################
# Get the initial state (psi_ref)

@cudaq.kernel
def initial_state(qubits_num:int):
    qubits = cudaq.qvector(qubits_num)
    h(qubits)

state = cudaq.get_state(initial_state, qubits_num)

#print(state)
###############################################

# Circuit to compute the energy gradient with respect to the pool
@cudaq.kernel
def grad(state:cudaq.State, ham_word:list[cudaq.pauli_word], ham_coef: list[complex], init_gamma:float):
    q = cudaq.qvector(state)

    for i in range(len(ham_coef)):
        exp_pauli(init_gamma * ham_coef[i].real, q, ham_word[i])


# The qaoa circuit using the selected pool operator with max gradient

@cudaq.kernel
def kernel_qaoa(qubits_num:int, ham_word:list[cudaq.pauli_word], ham_coef:list[complex],
        mixer_pool:list[list[cudaq.pauli_word]], gamma:list[float], beta:list[float], num_layer:int):

    qubits = cudaq.qvector(qubits_num)

    h(qubits)


    for p in range(num_layer):
        for i in range(len(ham_coef)):
            exp_pauli(gamma[p] * ham_coef[i].real, qubits, ham_word[i])

        for word in mixer_pool[p]:
            exp_pauli(beta[p], qubits, word)

# Begining od ADAPT-QAOA iteration

beta = []
gamma = []

mixer_pool = []
mixer_pool_str = []
layer=[]

istep = 1
while True:

    print('Step: ', istep)

    #####################
    # compute the gradient and find the mixer pool with large values.
    # If norm is below the predefined threshold, stop calculation

    gradient_vec = []
    for op in grad_op:
        op = op * -1j
        gradient_vec.append(cudaq.observe(grad, op , state, ham_word, ham_coef, init_gamma).expectation())

    # Compute the norm of the gradient vector
    norm = np.linalg.norm(np.array(gradient_vec))
    print('Norm of the gradient: ', norm)

    if norm <= threshold:
        print('\n', 'Final Result: ', '\n')
        print('Final mixer_pool: ', mixer_pool_str)
        print('Number of layers: ', len(layer))
        print('Number of mixer pool in each layer: ', layer)
        print('Final Energy: ', result_vqe.fun)

        break

    else:
        temp_pool = []
        tot_pool = 0

        max_grad = np.max(np.abs(gradient_vec))

        for i in range(len(pools)):
            if np.abs(gradient_vec[i]) >= max_grad:
                tot_pool += 1
                temp_pool.append(pools[i])

        # Set the seed for the random number generator
        # This ensures that the random choices are reproducible
        # in each step of the iteration.
        random.seed(42)

        layer.append(1)
        random_mixer = random.choice(temp_pool)

        pool_added = []
        pool_added_str = []
        for term in random_mixer:
            pool_added.append(cudaq.pauli_word(term.get_pauli_word(qubits_num)))
            pool_added_str.append(term.get_pauli_word(qubits_num))

        mixer_pool.append(pool_added)
        mixer_pool_str.append(pool_added_str)


        print('Mixer pool at step', istep)
        print(mixer_pool_str)

        num_layer = len(layer)
        print('Number of layers: ', num_layer)
        print(layer)
        beta_count = layer[num_layer-1]
        init_beta = [0.0] * beta_count
        beta = beta + init_beta
        gamma = gamma + [init_gamma]
        theta = gamma + beta
        print("ok")
        def cost(theta):

            theta = theta.tolist()
            gamma = theta[:num_layer]
            beta = theta[num_layer:]


            energy = cudaq.observe(kernel_qaoa, ham, qubits_num, ham_word, ham_coef,
                                    mixer_pool, gamma, beta, num_layer).expectation()
            #print(energy)
            return energy

        result_vqe = minimize(cost, theta, method='BFGS', jac='2-point', tol=1e-5)
        print('Result from the step ', istep)
        print('Optmized Energy: ', result_vqe.fun)

        dE = np.abs(result_vqe.fun - E_prev)
        E_prev = result_vqe.fun
        theta = result_vqe.x.tolist()
        erg = result_vqe.fun

        print('dE= :', dE)
        gamma=theta[:num_layer]
        beta=theta[num_layer:]

        if dE <= e_stop:
            print('\n', 'Final Result: ', '\n')
            print('Final mixer_pool: ', mixer_pool_str)
            print('Number of layers: ', len(layer))
            print('Number of mixer pool in each layer: ', layer)
            print('Final Energy= ', erg)

            break

        else:

            # Compute the state of this current step for the gradient
            state = cudaq.get_state(kernel_qaoa, qubits_num, ham_word, ham_coef,mixer_pool, gamma, beta, num_layer)
            #print('State at step ', istep)
            #print(state)
            istep+=1
            print('\n')

print('\n', 'Sampling the Final ADAPT QAOA circuit', '\n')
# Sample the circuit
count=cudaq.sample(kernel_qaoa, qubits_num, ham_word, ham_coef,mixer_pool, gamma, beta, num_layer, shots_count=5000)
print('The most probable max cut: ', count.most_probable())
print('All bitstring from circuit sampling: ', count)
