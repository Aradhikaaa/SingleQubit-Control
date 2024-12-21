# This is the RL environment

import math
import cmath
import numpy as np
from scipy.linalg import expm
import os
os.system("clear")
import sys

print(sys.version)

# Defining pauli spin matrices

sx = 1/2 * np.mat([[0, 1],[ 1, 0]], dtype=complex)
sy = 1/2 * np.mat([[0, -1j],[1j, 0]], dtype=complex)
sz = 1/2 * np.mat([[1, 0],[0, -1]], dtype=complex)

def hamiltonian(j):
    """
    A function that calculates the Hamiltonian value based on the input parameter j.
    
    Parameters:
    j (int or float): The input parameter used in the Hamiltonian calculation.

    Returns:
    numpy.array: The calculated Hamiltonian value.
    """
    J = 4
    H = (j) * J * sz + sx
    return H


psi_target = np.mat([[1],[0]], dtype=complex) # Target state is ket(0)
psi_0 = np.mat([[0],[1]], dtype=complex) # Initial state is ket(1)
#print(psi_target)
#print(psi_0)
N=10
dt = 2 * np.pi/N # Time step
print(dt)
def phase2(z):
    '''
    Function that calculates the phase angle of a cpmplex number z.
    return phase angle in [0, 2pi] (radian)
    '''
    phase = cmath.phase(z)
    if phase < 0:
        phase += 2*math.pi
    return phase


def state_to_lattice_point(state):
    '''
    Function that converts a quantum state to a lattice point.
    return the list [theta_i, phi_i]
    Note: phi = 0 or 2pi are the same
    '''
    # angular resolution of the mesh grid.

    Dtheta = np.pi/30 
    Dphi = np.pi/30

    if state[0,0] == 0: # state[0,0] is the 1st element of the state vector
        ## Special case 1: [0, 1]
        theta, phi = math.pi, 0
    else:
        conj = state[0,0].conj()
        state_reg = state * (conj/abs(conj)) # This makes the 1st element of the state vector to be real
        if (state_reg[0,0].real)>= 1:
            # Unitary should preserve norm
            theta, phi = 0, 0
        else: 
            # print(state_reg[0,0].imag)     # this should be 0
            theta = 2 * math.acos(state_reg[0,0].real)
            # state_reg[1,0]/sin(theta/2) = cos(phi) + i sin(phi)
            if theta == 0:
                ## Special case 2: [1, 0]
                phi = 0
            else:
                phi = phase2(state_reg[1,0]/math.sin(theta/2))  #force the phase of the first elements to be 0.

    theta_i = round(theta/Dtheta)
    phi_i = round(phi/Dphi)
    if phi_i == round(2*math.pi/Dphi): # phi_i corresonds to the maximum value of phi which is 2pi, then we set it to 0
        phi_i = 0
    return [theta_i, phi_i]

class Maze:
# qubit in the Bloch Maze
    def __init__(self):
        self.action_space = ['0', '1']
        self.n_actions = len(self.action_space)
        self._build_maze()

    def _build_maze(self):
        self.state = psi_0

    def reset(self):
        self.state = psi_0
        self.counter = 0
        # print(dt)
        return state_to_lattice_point(self.state)

    def step(self, action):

        if action == 0:
            U = expm(-(1j) * hamiltonian(0) * dt)
        elif action == 1:
            U = expm(-(1j) * hamiltonian(1) * dt)

        self.state = U.dot(self.state)
        self.counter += 1

        s_ = self.state
        fidelity = (abs(s_.conj().T.dot(psi_target)[0,0]))**2
        error = 1-fidelity

        if error < 10e-3:
            reward = 5000
            done = True
            s_lattice = 'terminal'
        else:
            reward = -1*(error>=0.5) + 10*(error<0.5) + 100*(error<0.1)
            #reward = 10*(error<0.5) + 100*(error<0.1)
            done = (self.counter >= 2*np.pi/dt)
            s_lattice = state_to_lattice_point(s_)
        return s_lattice, reward, done, fidelity
    
    
    
    
    
    
    
    
    
    
    
