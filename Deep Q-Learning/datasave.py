import os
import numpy as np
from Env import SwapEnv
from net import DeepQNetwork
import csv

os.system("clear")

C = 1.0
N = 4
dt = 2 * np.pi / N
print("Number of steps:", N)
print("Number of time steps :", dt)

env = SwapEnv(C=C, dt=dt, N=N)
RL = DeepQNetwork(env.action_space.n, env.observation_space.shape[0],
                  learning_rate=0.01,
                  reward_decay=0.9,
                  e_greedy=0.99,
                  replace_target_iter=200,
                  memory_size=2000,
                  e_greedy_increment=0.001)

step = 0
fid_max = 0

# Lists to store actions, corresponding magnetic field configurations, and corresponding states.
actions_per_episode = []  # List to store actions for each episode
episode_data = []  # List to store episode fidelity, actions, and observations

env.reset()
initial_state = env.state.copy()
print(f"Initial State: {initial_state}")

for episode in range(500):
    observation = env.reset()
    episode_actions = []  # Store actions for the current episode
    episode_fidelity = []  # Store fidelity for the current episode

    for i in range(N):
        action = RL.choose_action(observation)
        observation_, reward, done, info = env.step(action, i)
        RL.store_transition(observation, action, reward, observation_)

        if (step > 500) and (step % 5 == 0):
            RL.learn()

        observation = observation_

        episode_actions.append(action)  # Record the action for this episode

        # Calculate fidelity at each step
        fidelity = np.abs(np.dot(np.conj(env.target_state), env.state))**2
        episode_fidelity.append(fidelity)

        if done:
            break

        step += 1

    # Calculate and track the maximum fidelity
    final_state = env.state.copy()
    fidelity = episode_fidelity[-1]  # Get final fidelity for this episode
    print("Episode: ", episode, "Fidelity =", fidelity, "Final state:", final_state)

    # Append the actions and fidelity of the current episode to the list
    actions_per_episode.append(episode_actions)
    episode_data.append({
        'Episode': episode,
        'Fidelity': fidelity,
        'Actions': episode_actions,
        'Observations': episode_fidelity
    })

    if episode >= 490:
        if fidelity > fid_max:
            fid_max = np.copy(fidelity)
            final_state = env.state.copy()

# Display actions taken in the last 10 episodes
print("----------")
print(f"Final State: {np.round(final_state, decimals=1)}")
print("----------")
print('Final fidelity =', fid_max)

print("Actions taken in the last 10 episodes:")
for i in range(10):
    episode_index = 499 - i  # Get the last 10 episodes
    if episode_index >= 0:  # Check to ensure we don't go out of bounds
        print(f"Episode {episode_index}: Actions: {actions_per_episode[episode_index]}")

# Write to CSV
csv_file_path = 'N=4.csv'  # Specify your desired file name
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=['Episode', 'Fidelity', 'Actions', 'Observations'])
    writer.writeheader()
    
    for data in episode_data:
        writer.writerow({
            'Episode': data['Episode'],
            'Fidelity': data['Fidelity'],
            'Actions': ' '.join(map(str, data['Actions'])),  # Convert actions to a space-separated string
            'Observations': ' '.join(map(str, data['Observations']))  # Convert observations to a space-separated string
        })

print(f"Data saved to {csv_file_path}")
