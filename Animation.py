#This is the code for the visualization of the Bloch sphere trajectory and the pulse sequence for the state preparation using Q-Learning
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# Given states: theta and phi values for Bloch sphere evolution

# State and Action sequence before learning
#states = [[30, 0], [24, 15], [22, 35], [23, 29], [21, 24], [17, 22], [18, 45], [24, 46], [29, 9], [24, 15], [18, 15]]
#actions = [0, 1, 1, 0, 1, 0, 1, 0, 0, 0]

# State and Action sequence after learning
states = [[30, 0], [24, 15], [18, 15], [12, 15], [6, 15], [0, 0]]  
actions = [0, 0, 0, 0, 0] 

# Scale theta and phi values
theta_max = 30  # maximum theta in degrees
phi_max = 60  # maximum phi in degrees

scaled_states = []
for theta, phi in states:
    scaled_theta = np.deg2rad(theta) / np.deg2rad(theta_max) * np.pi
    scaled_phi = np.deg2rad(phi) / np.deg2rad(phi_max) * (2 * np.pi)
    scaled_states.append((scaled_theta, scaled_phi))

# Convert spherical coordinates to Cartesian coordinates for Bloch sphere
def spherical_to_cartesian(theta, phi):
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return x, y, z

# Calculate the Cartesian coordinates for Bloch sphere trajectory
cartesian_coords = [spherical_to_cartesian(theta, phi) for theta, phi in scaled_states]

# X data for the pulse sequence
x_data = np.arange(11)  # X-axis with points 0, 1, 2,...,10

# Use the original actions without clipping
y_data = np.array(actions)

# Extend actions to 11 points using the last value if there are fewer actions
num_actions = len(actions)
if num_actions < 11:
    y_data = np.pad(y_data, (0, 11 - num_actions), 'edge')

# Set up figure with two subplots (side by side)
fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(122, projection='3d')  # Bloch sphere
ax2 = fig.add_subplot(121)  # Pulse sequence

# Set initial pulse sequence axis limits
ax2.set_ylim([min(y_data) - 5, max(y_data) + 5])  # Set limits dynamically based on action values
ax2.set_xlim([0, 10])
ax2.set_box_aspect(0.99)

# Initialize plots for the pulse sequence
line, = ax2.plot([], [], lw=2, color='blue')
pulse_marker, = ax2.plot([], [],)

# Update function for the combined animation
def update(frame):
    # Update Bloch sphere
    ax1.clear()

    # Remove axis scales but keep X, Y, Z labels
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_zticks([])

    ax1.set_xlim([-1, 1])
    ax1.set_ylim([-1, 1])
    ax1.set_zlim([-1, 1])

    # Draw the Bloch sphere with very low opacity (translucent)
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax1.plot_surface(x, y, z, color='lightblue', alpha=0.2, rstride=5, cstride=5, edgecolor='gray')

    # Plot the trajectory with increased visibility
    trajectory = np.array(cartesian_coords[:frame + 1])
    ax1.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], color='red', linewidth=4, marker='o', markersize=8, markerfacecolor='red')

    # Plot the current point
    ax1.scatter(*cartesian_coords[frame], color='black', s=150)

    # Set simple labels for the axes
    ax1.set_xlabel('X', fontsize=12, labelpad=5)
    ax1.set_ylabel('Y', fontsize=12, labelpad=5)
    ax1.set_zlabel('Z', fontsize=12, labelpad=5)
    ax1.set_title('Bloch Sphere Trajectory', fontsize=16, pad=10)

    # Update Pulse sequence
    ax2.clear()  # Clear the axis for fresh plot
    ax2.set_ylim([-1.5,1.5])  # Set limits dynamically based on action values
    ax2.set_xlim([0, 10])

    # Plot the pulse sequence as a rectangular step function
    ax2.step(x_data[:frame + 1], y_data[:frame + 1], lw=2, color='blue', where='post')  # Step pulse line
    ax2.plot(x_data[frame], y_data[frame])  # Current pulse point

    # Set labels and title
    ax2.set_title("Pulse Profile", fontsize=14)
    ax2.set_xlabel(r'Time Step $i$', fontsize=14)
    ax2.set_ylabel(r'$J_i$', fontsize=14)

# Create the combined animation
ani = animation.FuncAnimation(fig, update, frames=len(cartesian_coords), interval=1)

# Add a common title in the middle of the plot
fig.suptitle('After Learning!', fontsize=18, fontweight='bold', y=0.95)

# Save the animation as a video
ani.save('After Learning.mp4', writer='ffmpeg', fps=6.75)
