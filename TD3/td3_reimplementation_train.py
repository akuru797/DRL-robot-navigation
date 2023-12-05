import stat
from turtle import st
import tensorflow as tf
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from replay_buffer import ReplayBuffer
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

class Critic(nn.Module):
    # Single Critic network!
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.layer1a = nn.Linear(self.state_size, 800)
        self.layer2a = nn.Linear(800, 600)
        self.layer2b = nn.Linear(self.action_size, 600)

        # Equation (6) - combined layer
        # Unclear about this implementation though. - note for final paper!
        # Combined FC layer is not a common term, so originally thought it looked something like below:
        # OLD ==> self.layer3 = self.layer2a + self.layer2b # Plus bias?
        self.layer3 = nn.ReLU()
        self.layer4 = nn.Linear(600, 1) # output layer
    
    def forward(self, s, a):
        # s = state
        # a = action
        s = self.flatten(s)
        a = self.flatten(a)
        Ls = F.relu(self.layer1a(s))
        F.layer2a(out)
        F.layer2b(a)
        out_s = torch.mm(Ls, self.layer2a.weight.data.t())
        out_a = torch.mm(a, self.layer2a.weight.data.t())
        out = self.layer3(out_s + out_a + self.layer2a.bias.data)
        Q1 = self.layer4(out)

        return Q1

          
## RUNTIME PARAMETERS
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters for training = according to paper
num_episodes = 800 # number of episodes to use for training
max_steps = 500
v_max = 0.5 #m/s
w_max = 1 #rad/s

n_delayed_reward = 10 #steps
parameter_update_delay = 2 #episodes

min_dist_threshold = 1 # meter, distance to goal where we return success
action_size = 2
seed = 42

# Parameters that werent specified/unclear in paper
environment_dim = 20 # number of laser readings
robot_dim = 4 # dimension of state  - In paper it makes it seem like robot dimension is 2, relative distance and heading to local waypoint!
buffer_size = 1e6 # max size of buffer
batch_size = 40 # Size of the mini-batch

# notes to self
# Gaussian noise added to sensor and action values
# every time environment resets, obstacle locations, starts, and goals change
# Actor proposes set of possible actions given state
# Critic: Estimated value function, evaluates actions taken by actor based on given policy
# Set seed for experiment reproducibility

num_episodes = 0
state_size = robot_dim + environment_dim
# Create a replay buffer
replay_buffer = ReplayBuffer(buffer_size, seed)

# Initialize Actor and Critic Networks
actor = Actor(state_size, action_size).to(device)
critic1 = Critic(state_size, action_size).to(device)
critic2 = Critic(state_size, action_size).to(device)

# Target Networks
actor_target = Actor.load_state_dict(actor.state_dict())
critic1_target = Critic.load_state_dict(critic1.state_dict())
critic2_target = Critic.load_state_dict(critic2.state_dict())


env = GazeboEnv("multi_robot_scenario.launch", environment_dim)

# Optimizers
actor_optimizer = torch.optim.Adam(actor.parameters())
critic1_optimizer = torch.optim.Adam(critic1.parameters())
critic2_optimizer = torch.optim.Adam(critic2.parameters())
"""
1. Run the agent on the environment to collect training data per episode.
2. Compute expected return at each time step.
3. Compute the loss for the combined Actor-Critic model.
4. Compute gradients and update network parameters.
Repeat 1-4 until either success criterion or max episodes has been reached.
"""
t = 0
updatec1 = True
for t in range(num_episodes):
    state = env.reset()

    done = False
    t_episode = 0
    while (not done):
        # Within each episode
        a = actor(state) # select action 
        en = np.random.normal(size=[1,2]) # exploration noise sigma assumed to be 1

        # Scaling according to eqn 5
        v = v_max * (a(0) + 1)/2
        w = w_max * a(1)

        action = [v,w]
        next_state, reward, done, target = env.step(action)

        # Store transition in replay buffer
        replay_buffer.add(state, action, reward, done, next_state)
        state = next_state
        t_episode += 1
        t+=1

    if (replay_buffer.size > batch_size):
        ## FOR EVERY EPISODE - TRAIN
        # Sample mini-batch of transitions from buffer
        (
            batch_states,
            batch_actions,
            batch_rewards,
            batch_dones,
            batch_next_states,
        ) = replay_buffer.sample_batch(batch_size)
        
        with torch.no_grad():
            a_tilda = actor(batch_states)
            Q1 = critic1_target(state, action)
            Q2 = critic2_target(state, action)
            Q = min(Q1, Q2)
            target_Q = reward + (1 - done) * gamma * Q

        # Update Actor Network
        # Compute critic loss
        current_Q1 = critic1(state, action)
        current_Q2 = critic2(state, action)
        critic1_loss = F.mse_loss(current_Q1, target_Q)
        critic2_loss = F.mse_loss(current_Q2, target_Q)
    
        
        # Update Targets
        if i_episode % parameter_update_delay:
                # Optimize Critic Networks
                if (updatec1):
                    critic1_optimizer.zero_grad()
                    critic1_loss.backward()
                    critic1_optimizer.step()
                else:
                    critic2_optimizer.zero_grad()
                    critic2_loss.backward()
                    critic2_optimizer.step()
                
                updatec1 = not updatec1 # switch which critic network gets updated

        # Optimize the actor 
        actor_optimizer.zero_grad()
        actor_loss.backward()
        sactor_optimizer.step()

        # Update the frozen target models
        if i_episode % 100
            actor_target.load_state_dict(actor.state_dict())
            critic1_target.load_state_dict(critic1.state_dict())
            critic2_target.load_state_dict(critic2.state_dict())
