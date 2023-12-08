import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib as plt
from velodyne_env import GazeboEnv


class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        # input is size 23
        # FC800, ReLu, FC600, ReLu, FC2, tanh
        self.actor_stack = nn.Sequential(
            nn.Linear(self.state_size, 800),
            nn.ReLU(),
            nn.Linear(800, 600), 
            nn.ReLU(),
            nn.Linear(600,self.action_size),
            nn.Tanh()
        )

    def forward(self, s):
        # s is input data, state
        action = self.actor_stack(s)
        # (Q) should this return a distribution, like here: https://github.com/yc930401/Actor-Critic-pytorch/blob/master/Actor-Critic.py
        return action


# Set the parameters for the implementation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cuda or cpu
seed = 0  # Random seed number
max_ep = 500  # maximum number of steps per episode
file_name = "TD3_velodyne"  # name of the file to load the policy from
directory = "./pytorch_models/td3_reimpl/"
model_path = "%s/%s_actor.pth" % (directory, file_name)

# Create the testing environment
environment_dim = 20
robot_dim = 4
env = GazeboEnv("multi_robot_scenario.launch", environment_dim)
time.sleep(5)
torch.manual_seed(seed)
np.random.seed(seed)
state_dim = environment_dim + robot_dim
action_dim = 2

# Create the network
network = Actor(state_dim, action_dim)
try:
    network.load_state_dict(torch.load(model_path))
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
collision_count = 0
while i_episode < num_eval_episodes:
    s = torch.Tensor(state.reshape(1,-1)).to(device)
    action = network(s).cpu().data.numpy().flatten()

    # Update action to fall in range [0,1] for linear velocity and [-1,1] for angular velocity
    a_in = [(action[0] + 1) / 2, action[1]]
    next_state, reward, done, target = env.step(a_in)
    if reward < -90:
        collision_count += 1

    done = 1 if episode_timesteps + 1 == max_ep else int(done)

    # On termination of episode
    if done:
        state = env.reset()
        done = False
        episode_length[i_episode] = episode_timesteps
        episode_timesteps = 0
        cumulative_reward = 0
        i_episode += 1
    else:
        state = next_state
        episode_timesteps += 1
        cumulative_reward += reward

# Plots / results
colstr = "Collision rate is " + str(collision_count/num_eval_episodes)
print(colstr)

# Cumulative Rewards
fig = plt.figure(figsize=(10,5))
plt.hist(rewards, bins='auto')
titlestr = "Episode Length over " + str(num_eval_episodes) + " evaluation episodes"
plt.title(titlestr, bins='auto')
plt.show()

# Episode Length
fig = plt.figure(figsize=(10,5))
plt.hist(episode_length)
titlestr = "Episode Length over " + str(num_eval_episodes) + " evaluation episodes"
plt.title(titlestr)
plt.show()