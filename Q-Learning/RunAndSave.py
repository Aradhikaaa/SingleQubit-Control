# This runs the code and saves the data of evolving state and action sequence for the episode
import csv
import Environment as env
from function import QlearningTable
import numpy as np
import os
os.system("clear")

# Initialize the environment and Q-learning table
env = env.Maze()
RL = QlearningTable(actions=list(range(env.n_actions)))

# Parameters
ep_max = 15000
fidelity = np.zeros(ep_max)
best_fid = 0

# Track all episodes' data
episode_data = []

for episode in range(ep_max):
    observation = env.reset()
    actions_taken = []
    observations_taken = [observation]
    
    while True:
        action = RL.choose_action(str(observation),episode)
        observation_, reward, done, fid = env.step(action)
        RL.learn(str(observation), action, reward, str(observation_))
        observation = observation_
        actions_taken.append(action)
        observations_taken.append(observation_)
        
        if done:
            # Store fidelity value for the current episode
            fidelity[episode] = fid
            
            # Append episode data to list
            episode_data.append({
                'Episode': episode,
                'Fidelity': fid,
                'Actions': actions_taken,
                'Observations': observations_taken
            })
            
            # Check if this episode has the highest fidelity among the last 100 episodes
            if episode >= ep_max - 101:
                print(f'Episode {episode}: Fidelity = {fid}')
                if fid > best_fid:
                    best_fid = fid
                    best_actions = actions_taken.copy()
                    best_observations = observations_taken.copy()
            break

# Print the final fidelity values
print('Final fidelity of the last 100 episodes:', best_fid)

# Print the action sequence of the episode with the highest fidelity among the last 10 episodes
print('Action sequence of the episode with highest fidelity:')
print(best_actions)

# Print the evolved states (observations) of the episode with the highest fidelity among the last 10 episodes
print('Evolved states (observations) of the episode with highest fidelity:')
for i, obs in enumerate(best_observations):
    print(f'Step {i}: {obs}')

# CSV file name
csv_filename = 'N10.csv'

# Write all episode data to CSV file
with open(csv_filename, mode='w', newline='') as csv_file:
    fieldnames = ['Episode', 'Fidelity', 'Actions', 'Observations']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    
    # Write header
    writer.writeheader()
    
    # Write data rows
    for data in episode_data:
        writer.writerow({
            'Episode': data['Episode'],
            'Fidelity': data['Fidelity'],
            'Actions': data['Actions'],
            'Observations': data['Observations']
        })

print(f'CSV file "{csv_filename}" has been created with episode data.')

#RL.display_q_table()