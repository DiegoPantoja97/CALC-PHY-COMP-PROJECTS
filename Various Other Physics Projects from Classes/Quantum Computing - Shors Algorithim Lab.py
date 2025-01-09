#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
import matplotlib.pyplot as plt
from math import pi, cos, sin
from fractions import Fraction
from math import gcd
from collections import Counter



from qiskit.visualization import array_to_latex
from qiskit.primitives import StatevectorEstimator, StatevectorSampler


from qiskit.quantum_info import SparsePauliOp
from qiskit.quantum_info import Statevector,random_statevector
from qiskit.quantum_info.operators import Operator, Pauli


from qiskit.primitives import Sampler
from qiskit.visualization import plot_histogram
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import QFT

from qiskit.circuit.library import HGate, CXGate
from IPython.display import display, Latex
from qiskit.circuit import Parameter



"""Part 1: Phase Estimation - Procedure  1 QUBIT"""


theta = 0.7  # Can be changed to any value between 0 and 1

qc = QuantumCircuit(2, 1)

# Prepare eigenvector, which is the |1> state

qc.x(1)
qc.barrier()

# Implement the estimation procedure
qc.h(0)
qc.cp(2 * pi * theta, 0, 1)
qc.h(0)
qc.barrier()

# Perform the final measurement
qc.measure(0, 0)

# Draw the circuit
display(qc.draw(output = "mpl"))

display(Sampler().run(qc).result().quasi_dists[0])

display({   # Calculate predicted results
    0: cos(pi * theta) ** 2,
    1: sin(pi * theta) ** 2
})


# In[19]:


"""Part 2: Phase Estimation - Procedure  2 QUBIT"""

theta = 0.7
qc = QuantumCircuit(3, 2)

# Prepare the eigenvector
qc.x(2)
qc.barrier()

# The initial Hadamard gates
qc.h(0)
qc.h(1)
qc.barrier()

# The controlled unitary gates
qc.cp(2 * pi * theta, 0, 2)
qc.cp(2 * pi * (2 * theta), 1, 2)
qc.barrier()

# An implementation of the inverse of the two-qubit QFT
qc.swap(0, 1)
qc.h(0)
qc.cp(-pi / 2, 0, 1)
qc.h(1)
qc.barrier()

# And finally the measurements
qc.measure([0, 1], [0, 1])
display(qc.draw(output = "mpl"))

result = Sampler().run(qc).result()
display(plot_histogram(result.quasi_dists))


# In[41]:


"""Part 3 - QFT and IQFT """

from qiskit.circuit.library import QFT
num_qubits = 4
display(QFT(num_qubits).decompose().draw(output = "mpl"))
display(QFT(num_qubits, inverse=True).decompose().draw(output = "mpl") )


# In[57]:


""" Part 4 - Phase Estimation Circuit """

theta = 0.7
m = 4  # Number of control qubits

control_register = QuantumRegister(m, name="Control")
target_register = QuantumRegister(1, name="|ψ>")
output_register = ClassicalRegister(m, name="Result")
qc = QuantumCircuit(control_register, target_register, output_register)

# Prepare the eigenvector |ψ>
qc.x(target_register)
qc.barrier()

# Perform phase estimation
for index, qubit in enumerate(control_register):
    qc.h(qubit)
    for _ in range(2**index):
        qc.cp(2 * pi * theta, qubit, target_register)
qc.barrier()

# Do inverse quantum Fourier transform
qc.compose(
    QFT(m, inverse=True),
    inplace=True
)

# Measure everything
qc.measure(range(m), range(m))
display(qc.draw(output = 'mpl'))

result = Sampler().run(qc).result()
display(plot_histogram(result.quasi_dists))

most_probable = max(result.quasi_dists[0], key=result.quasi_dists[0].get)

print(f"Most probable output: {most_probable}")
print(f"Estimated theta: {most_probable/2**m}")



# In[59]:


N = 21
a = 17
max_power = 12

print("k \t a^k \n")
for k in range(1, max_power + 1):
    print(
        "%2d \t %2d" % (k, a**k % N)
    )  # The % operation computes the remainder modulo N







# In[71]:


""" Part 5 - Order Finding"""


"""Function Description: Two numbers are coprime if their greatest common divisor (GCD) is 1
def c_amod15(a): This function implements a controlled unitary operation implements a controlled modular multiplication operation 
on a quantum circuit. Its goal is to simulate the effect of multiplying by a modulo 15 on a quantum state
    """

""" Input: A number (integer) that is coprime to 15 The input quantum state is represented in 4 qubits, which can encode integers from 0-15 in Binary"""

""" Output: A controlled unitary operation applies U conditionally on a control qubit. """"

"""EXAMPLE:  |x> ---> |(2*x) mod15> """

def c_amod15(a):
    """
    Controlled multiplication by a mod 15.
    This is hard-coded for simplicity.
    """
    if a not in [2, 4, 7, 8, 11, 13]:
        raise ValueError("'a' must not have common factors with 15")
    U = QuantumCircuit(4)
    if a in [2, 13]:
        U.swap(2, 3)
        U.swap(1, 2)
        U.swap(0, 1)
    if a in [7, 8]:
        U.swap(0, 1)
        U.swap(1, 2)
        U.swap(2, 3)
    if a in [4, 11]:
        U.swap(1, 3)
        U.swap(0, 2)
    if a in [7, 11, 13]:
        for q in range(4):
            U.x(q)
    U = U.to_gate()
    U.name = f"{a} mod 15"
    c_U = U.control()
    return c_U



def phase_estimation(
        controlled_operation: QuantumCircuit,
        psi_prep: QuantumCircuit,
        precision: int
    ):
    """
    Carry out phase estimation on a simulator.
    Args:
        controlled_operation: The operation to perform phase estimation on,
                              controlled by one qubit.
        psi_prep: Circuit to prepare |ψ>
        precision: Number of counting qubits to use
    Returns:
        float: Best guess for phase of U|ψ>
    """
    control_register = QuantumRegister(precision)
    output_register = ClassicalRegister(precision)

    target_register = QuantumRegister(psi_prep.num_qubits)
    qc = QuantumCircuit(control_register, target_register, output_register)

    # Prepare |ψ>
    qc.compose(psi_prep,
               qubits=target_register,
               inplace=True)

    # Do phase estimation
    for index, qubit in enumerate(control_register):
        qc.h(qubit)
        for _ in range(2**index):
            qc.compose(
                controlled_operation,
                qubits=[qubit] + list(target_register),
                inplace=True,
            )

    qc.compose(
        QFT(precision, inverse=True),
        qubits=control_register,
        inplace=True
    )

    qc.measure(control_register, output_register)

    measurement = Sampler().run(qc, shots=1).result().quasi_dists[0].popitem()[0]
    return measurement / 2**precision

psi_prep = QuantumCircuit(4)
psi_prep.x(0)
display(psi_prep.draw(output = 'mpl'))

a = 2
N = 15

FACTOR_FOUND = False
ATTEMPT = 0
while not FACTOR_FOUND:
    ATTEMPT += 1
    print(f"\nAttempt {ATTEMPT}")

    phase = phase_estimation(
        c_amod15(a),
        psi_prep,
        precision=8
    )
    frac = Fraction(phase).limit_denominator(N)
    r = frac.denominator
    if phase != 0:
        # Guess for a factor is gcd(x^{r/2} - 1 , 15)
        guess = gcd(a ** (r // 2) - 1, N)
        if guess not in [1, N] and (N % guess) == 0:
            # Guess is a factor!
            print(f"Non-trivial factor found: {guess}")
            FACTOR_FOUND = True


# In[32]:


""" Lab 3 - Assignment: Take the Above Circuit and Implement a c_amod12(a) via hard-coding as described above for c_amode15(a) """


def c_amod12(a):
    # Step 1 Identify Coprime integers: gcd(a,12) = 1
    if a not in [5, 7, 11]:
        raise ValueError("'a' must not have common factors with 12")
    
    U = QuantumCircuit(4)

    # Swapping Qubits to Encode Modular Multiplication
    # Define swap patterns for specific a values
    if a in [5, 7]:
        U.swap(2, 3)
        U.swap(1, 2)
        U.swap(0, 1)
    if a in [11]:
        U.swap(0, 2)
        U.swap(1, 3)
    if a in [7, 11]:
        for q in range(4):
            U.x(q)
    
    U = U.to_gate()
    U.name = f"{a} mod 12"
    c_U = U.control()
    return c_U


def phase_estimation(
        controlled_operation: QuantumCircuit,
        psi_prep: QuantumCircuit,
        precision: int
    ):
    """
    Carry out phase estimation on a simulator.
    Args:
        controlled_operation: The operation to perform phase estimation on,
                              controlled by one qubit.
        psi_prep: Circuit to prepare |ψ>
        precision: Number of counting qubits to use
    Returns:
        float: Best guess for phase of U|ψ>
    """
    control_register = QuantumRegister(precision)
    output_register = ClassicalRegister(precision)

    target_register = QuantumRegister(psi_prep.num_qubits)
    qc = QuantumCircuit(control_register, target_register, output_register)

    # Prepare |ψ>
    qc.compose(psi_prep,
               qubits=target_register,
               inplace=True)

    # Do phase estimation
    for index, qubit in enumerate(control_register):
        qc.h(qubit)
        for _ in range(2**index):
            qc.compose(
                controlled_operation,
                qubits=[qubit] + list(target_register),
                inplace=True,
            )

    qc.compose(
        QFT(precision, inverse=True),
        qubits=control_register,
        inplace=True
    )

    qc.measure(control_register, output_register)

    measurement = Sampler().run(qc, shots=1).result().quasi_dists[0].popitem()[0]
    return measurement / 2**precision

psi_prep = QuantumCircuit(4)
psi_prep.x(0)

a = 7
N = 12

FACTOR_FOUND = False
ATTEMPT = 0
while not FACTOR_FOUND:
    ATTEMPT += 1
    print(f"\nAttempt {ATTEMPT}")

    phase = phase_estimation(
        c_amod12(a),
        psi_prep,
        precision=8
    )
    frac = Fraction(phase).limit_denominator(N)
    r = frac.denominator
    if phase != 0:
        # Guess for a factor is gcd(x^{r/2} - 1 , 15)
        guess = gcd(a ** (r // 2) - 1, N)
        if guess not in [1, N] and (N % guess) == 0:
            # Guess is a factor!
            print(f"Non-trivial factor found: {guess}")
            FACTOR_FOUND = True


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




