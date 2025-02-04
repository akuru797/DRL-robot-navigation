import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from velodyne_env import GazeboEnv


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.layer_1 = nn.Linear(state_dim, 800)
        self.layer_2 = nn.Linear(800, 600)
        self.layer_3 = nn.Linear(600, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, s):
        s = F.relu(self.layer_1(s))
        s = F.relu(self.layer_2(s))
        a = self.tanh(self.layer_3(s))
        return a


# TD3 network
class TD3(object):
    def __init__(self, state_dim, action_dim):
        # Initialize the Actor network
        self.actor = Actor(state_dim, action_dim).to(device)

    def get_action(self, state):
        # Function to get the action from the actor
        state = torch.Tensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def load(self, filename, directory):
        # Function to load network parameters
        self.actor.load_state_dict(
            torch.load(
                "%s/%s_actor.pth" % (directory, filename),
                map_location=torch.device("cpu"),
            )
        )


# Set the parameters for the implementation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cuda or cpu
seed = 0  # Random seed number
max_ep = 500  # maximum number of steps per episode
file_name = "TD3_velodyne"  # name of the file to load the policy from


# Create the testing environment
environment_dim = 20
robot_dim = 4
env = GazeboEnv("multi_robot_scenario.launch", environment_dim)
time.sleep(25)
torch.manual_seed(seed)
np.random.seed(seed)
state_dim = environment_dim + robot_dim
action_dim = 2

# Create the network
network = TD3(state_dim, action_dim)
try:
    network.load(file_name, "./pytorch_models")
except:
    raise ValueError("Could not load the stored model parameters")

done = False
episode_timesteps = 0
state = env.reset()

# Begin the testing loop
num_eval_episodes = 20
i_episode = 0
episode_length = np.zeros(num_eval_episodes)
rewards = np.zeros(num_eval_episodes)
cumulative_reward = 0
collision_counter = 0
while i_episode < num_eval_episodes:
    action = network.get_action(np.array(state))

    # Update action to fall in range [0,1] for linear velocity and [-1,1] for angular velocity
    a_in = [(action[0] + 1) / 2, action[1]]
    next_state, reward, done, target = env.step(a_in)
    done = 1 if episode_timesteps + 1 == max_ep else int(done)
    if reward == -100:
        collision_counter += 1

    cumulative_reward += reward
    # On termination of episode
    if done:
        state = env.reset()
        done = False
        episode_length[i_episode] = episode_timesteps
        rewards[i_episode] = cumulative_reward
        episode_timesteps = 0
        cumulative_reward = 0
        notestr = "Episode #" + str(i_episode) + " completed."
        print(notestr)
        i_episode += 1
    else:
        state = next_state
        episode_timesteps += 1

collision_percentage = (collision_counter / num_eval_episodes) * 100

# Plots
# Cumulative Rewards
fig = plt.figure(figsize=(10, 5))
plt.hist(rewards, bins="auto", rwidth=0.9)
titlestr = "Rewards over " + str(num_eval_episodes) + " evaluation episodes"
plt.title(titlestr)
plt.savefig("Rewards.png")

# Episode Length
fig = plt.figure(figsize=(10, 5))
plt.hist(episode_length, bins=50, rwidth=0.9)
titlestr = "Episode Length over " + str(num_eval_episodes) + " evaluation episodes"
plt.title(titlestr)
plt.savefig("Episode_Length.png")
plt.show()

print("The collision percentage is: " + str(collision_percentage) + " %")
