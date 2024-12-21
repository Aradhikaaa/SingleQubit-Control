# This is the code for the RL agent which has the Q-learning algorithm

import numpy as np
import pandas as pd
import os

os.system("clear")
class QlearningTable:
    def __init__(self,actions, alpha=0.0001, gamma=0.99, epsilon=0.3):

        self.actions=actions
        self.alpha=alpha #learning rate
        self.gamma=gamma #reward decay
        self.epsilon=epsilon #epilson greedy
        #self.numberEpisodes=numberEpisodes

        # this list stores sum of rewards in every learning episode
        #self.sumRewardsEpisode=[]
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)


    def choose_action(self,observation,episode):
        """
        Chooses the action based on the observation and current episode.

        Parameters:
            observation (str): The current observation.
            episode (int): The current episode number.

        Returns:
            action: The selected action based on the strategy.
        """
        self.check_state_exist(observation)
        # action selection

        #first 500 episodes we select completely random actions 
        if episode<500:
            action = np.random.choice(self.actions)

        randomNumber=np.random.uniform()

        #after 7000 episodes, we slowly start to decrease the epilson greedy parameter
        if episode>7000:
            self.epsilon=0.999*self.epsilon

        #if this condition is satisfied, we are exploring, that is, we select random actions
        if randomNumber<self.epsilon:
            action=np.random.choice(self.actions)
            

        #otherwise, we select the action with the highest Q value
        else:
            state_action = self.q_table.loc[observation, :]
            state_action = state_action.reindex(np.random.permutation(state_action.index))     # some actions have same value
            action = state_action.idxmax()
        return action
    
    def learn(self, s, a, r, s_):
        """
        s,s_ --string of state
        a --action
        r --reward
        """
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.alpha * (q_target - q_predict)

    def check_state_exist(self, state):
          """
          This function checks if the state exists in the q_table index. 
          If the state is not in the index, it creates a new row for the 
          state by concatenating a new state row to the q_table.
          """
          if state not in self.q_table.index:
            # Create a new row for the new state
            new_state_row = pd.DataFrame(
                [[0] * len(self.actions)],
                index=[state],
                columns=self.q_table.columns
            )
            # Concatenate the new state row to the q_table
            self.q_table = pd.concat([self.q_table, new_state_row])
    

    def display_q_table(self):
        """
        Displays the Q-table.
        """
        print("Q-Table:")
        print(self.q_table)
    