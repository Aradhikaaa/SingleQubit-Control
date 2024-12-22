import gym
import numpy as np
from scipy.linalg import expm
from gym import spaces
import os
os.system("clear")

class SwapEnv(gym.Env):
    def __init__(self, C, dt, N):
        super(SwapEnv, self).__init__()
        self.C = C
        self.dt = dt
        self.N = N  # Number of steps in the sequence
        
        # Pauli matrices
        self.sigma_x = np.array([[0, 1], [1, 0]])
        self.sigma_y = np.array([[0, -1j], [1j, 0]])
        self.sigma_z = np.array([[1, 0], [0, -1]])
        self.identity = np.eye(2)
        
        # Initial and target states
        self.initial_state = np.array([0,1], dtype=complex)
        self.target_state = np.array([1,0], dtype=complex)
        # Action space
        self.action_space = spaces.Discrete(2)
        
        # Observation space
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # Reset the environment to the initial state
        self.reset()


    
    def hamiltonian(self, j):
        H=4*j*self.sigma_z+self.sigma_x
        return H
    
    def step(self, action, step_count):
        # Get the Hamiltonian for the given action
        H = self.hamiltonian(action)
        U = expm(-1j * H * self.dt)  # Unitary operator for time evolution
        
        # Evolve the state
        self.state = np.dot(U, self.state)
        
        # Calculate reward based on fidelity
        fidelity = np.abs(np.dot(np.conj(self.target_state), self.state))**2
        reward = fidelity * 10 if fidelity < 0.999 else 2500
        done = fidelity >= 0.999
        
        # Convert the state to real and imaginary parts for observation
        observation = np.concatenate((self.state.real, self.state.imag))

        #observation = np.array([self.state[0].real, self.state[0].imag])
        
        return observation, reward, done, {}
    
    def reset(self):
        """
        Resets the environment to its initial state.

        Returns:
        numpy array: The initial state as a numpy array, with real and imaginary parts concatenated.
        """
        self.state = self.initial_state.copy()
        return np.concatenate((self.state.real, self.state.imag))
