import pennylane as qml
from qiskit import *
import matplotlib.pyplot as plt
from pennylane import numpy as np
from pyscf import gto, scf
from qiskit import QuantumCircuit, Aer, transpile
from qiskit.visualization import plot_histogram
from qiskit import QuantumCircuit, Aer, transpile, assemble
from qiskit.visualization import plot_histogram
from qiskit import QuantumCircuit, Aer, transpile, assemble, execute
from qiskit.visualization import plot_histogram
from qiskit.providers.aer.noise import NoiseModel, depolarizing_error
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from IPython.display import clear_output
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit_algorithms.optimizers import COBYLA, L_BFGS_B
from qiskit_algorithms.utils import algorithm_globals

from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier, VQC
from qiskit_machine_learning.algorithms.regressors import NeuralNetworkRegressor, VQR
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from qiskit_machine_learning.circuit.library import QNNCircuit
from qiskit.quantum_info import Statevector, DensityMatrix
from qiskit.visualization import circuit_drawer, plot_bloch_multivector, plot_circuit_layout
import matplotlib.pyplot as plt
from qiskit import Aer, execute
import streamlit as st
import numpy as np
from qiskit.visualization import plot_histogram
from qiskit import Aer, transpile, assemble, execute
from qiskit.providers.aer.noise import NoiseModel, depolarizing_error
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import seaborn as sns
# creation of hamilitonian operators -kinetic energy and potential energy
#H=K+V
# Where K is the kinetic energy operator and V is potential energy operator
#m- mass of electron
#h_bar -reduced planck constant
#n-number of electrons
#Z-atomic number
#a-screening constant
# we can adjust the specific of parameters in the function since we understand atoms are 3D but we assume 1D for now
#we create a function for both operators


def kinetic_energy_operator(n, h_bar=1.0, m=1.0):
    """
    Kinetic energy operator for n electrons in one dimension.
    """
    T = np.zeros((n, n))
    for i in range(n):
        T[i, i] = -0.5 * (h_bar**2) / m
    return T

def potential_energy_operator(n, Z=5, a=1.0, h_bar=1.0):
    """
    Potential energy operator for n electrons interacting with a nucleus.
    """
    V_nucleus = -Z / a
    V = np.zeros((n, n))
    for i in range(n):
        V[i, i] = V_nucleus
    return V

def hamiltonian_operator(n, h_bar=1.0, m=1.0, Z=5, a=1.0):
    """
    Hamiltonian operator for n electrons in one dimension.
    """
    T = kinetic_energy_operator(n, h_bar, m)
    V = potential_energy_operator(n, Z, a, h_bar)
    H = T + V
    return H

# Boron atom with 5 electrons and 1 unoccupied orbital
n_electrons = 5
H_boron = hamiltonian_operator(n_electrons)

print("Hamiltonian Matrix for Boron:")
print(H_boron)
