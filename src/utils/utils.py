from qibo import Circuit, gates
from qibo.optimizers import optimize
from qibo.quantum_info import infidelity
import numpy as np

# custom loss function, computes fidelity
def myloss(parameters, circuit, target):
    circuit.set_parameters(parameters)
    final_state = circuit().state()
    return infidelity(final_state, target)

nqubits = 6
dims = 2**nqubits
nlayers  = 2

# Create variational circuit
circuit = Circuit(nqubits)
for l in range(nlayers):
    circuit.add(gates.RY(qubit, theta=0.0) for qubit in range(nqubits))
    circuit.add(gates.CZ(qubit, qubit + 1) for qubit in range(0, nqubits - 1, 2))
    circuit.add(gates.RY(qubit, theta=0.0) for qubit in range(nqubits))
    circuit.add(gates.CZ(qubit, qubit + 1) for qubit in range(1, nqubits - 2, 2))
    circuit.add(gates.CZ(0, nqubits - 1))
circuit.add(gates.RY(qubit, theta=0.0) for qubit in range(nqubits))

# Optimize starting from a random guess for the variational parameters
x0 = np.random.uniform(0, 2 * np.pi, nqubits * (2 * nlayers + 1))
data = np.random.normal(0, 1, size=dims)

# perform optimization
best, params, extra = optimize(myloss, x0, args=(circuit, data), method='BFGS', options={'maxiter':10})

# set final solution to circuit instance
circuit.set_parameters(params)

from qibo import Circuit, gates

circuit = Circuit(2)
circuit.add(gates.X(0))
# Add a measurement register on both qubits
circuit.add(gates.M(0, 1))
# Execute the circuit with the default initial state |00>.
result = circuit(nshots=100)
print(result.frequencies(binary=True))

import itertools

vec_permutations = itertools.product([0, 1], repeat=3)
for permutation in vec_permutations:
    print(permutation)




