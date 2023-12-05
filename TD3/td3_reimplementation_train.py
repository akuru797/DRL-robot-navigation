import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from replay_buffer import ReplayBuffer
from velodyne_env import GazeboEnv
import os
import time

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
        Ls = F.relu(self.layer1a(s))
        self.layer2a(Ls)
        self.layer2b(a)
        out_s = torch.mm(Ls, self.layer2a.weight.data.t())
        out_a = torch.mm(a, self.layer2b.weight.data.t())
        out = self.layer3(out_s + out_a + self.layer2a.bias.data)
        Q1 = self.layer4(out)

        return Q1

def save(actor_state_dict, critic1_state_dict, critic2_state_dict, filename, directory):
    torch.save(actor_state_dict, "%s/%s_actor.pth" % (directory, filename))
    torch.save(critic1_state_dict, "%s/%s_critic1.pth" % (directory, filename))
    torch.save(critic2_state_dict, "%s/%s_critic2.pth" % (directory, filename))

## RUNTIME PARAMETERS
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
file_name = "TD3_velodyne"

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
gamma = 1
eval_freq = 5e3  # After how many steps to perform the evaluation
eval_episodes = 10

# notes to self
# Gaussian noise added to sensor and action values
# every time environment resets, obstacle locations, starts, and goals change
# Actor proposes set of possible actions given state
# Critic: Estimated value function, evaluates actions taken by actor based on given policy
# Set seed for experiment reproducibility

if not os.path.exists("./pytorch_models"):
    os.makedirs("./pytorch_models")


num_episodes = 800
state_size = robot_dim + environment_dim
# Create a replay buffer
replay_buffer = ReplayBuffer(buffer_size, seed)

# Initialize Actor and Critic Networks
actor = Actor(state_size, action_size).to(device)
critic1 = Critic(state_size, action_size).to(device)
critic2 = Critic(state_size, action_size).to(device)

# Target Networks
actor_target = Actor(state_size, action_size).to(device)
actor_target.load_state_dict(actor.state_dict())
critic1_target = Critic(state_size, action_size).to(device)
critic2_target = Critic(state_size, action_size).to(device)
critic1_target.load_state_dict(critic1.state_dict())
critic2_target.load_state_dict(critic2.state_dict())


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
timesteps_since_eval = 0
writer = SummaryWriter()
epoch = 0
time.sleep(15) # environment setup
for i_ep in range(num_episodes):
    state = env.reset()

    done = False
    t_episode = 0
    while (not done and t_episode <= max_steps):
        # Within each episode
        state_tensor = torch.Tensor(state.reshape(1,-1)).to(device)
        a = actor(state_tensor).cpu().data.numpy().flatten() # select action 
        en = np.random.normal(size=[1,2]) # exploration noise sigma assumed to be 1

        # Scaling according to eqn 5
        v = v_max * (a[0] + 1)/2
        w = w_max * a[1]

        action = [v,w]
        next_state, reward, done, target = env.step(action)

        # Store transition in replay buffer
        replay_buffer.add(state, action, reward, done, next_state)
        state = next_state
        t_episode += 1
        t+=1


    if (replay_buffer.size() > batch_size):
        # FOR EVERY EPISODE - TRAIN # is this true
        # Sample mini-batch of transitions from buffer
        (
            batch_state,
            batch_action,
            batch_reward,
            batch_done,
            batch_next_state,
        ) = replay_buffer.sample_batch(batch_size)

        batch_states = torch.Tensor(batch_state).to(device)
        batch_actions = torch.Tensor(batch_action).to(device)
        batch_rewards = torch.Tensor(batch_reward).to(device)
        batch_dones = torch.Tensor(batch_done).to(device)
        batch_next_states = torch.Tensor(batch_next_state).to(device)
        
        with torch.no_grad():
            # Compute Target action
            a_target = actor_target(batch_states)
            Q1 = critic1_target(batch_states, a_target)
            Q2 = critic2_target(batch_states, a_target)
            Q = torch.min(Q1, Q2)

            # Target Q
            target_Q = batch_rewards + (1 - batch_dones) * gamma * Q
        
        av_Q = torch.mean(target_Q)
        max_Q = torch.max(target_Q)

        # Optimize Critic Networks
        current_Q1 = critic1(batch_states, batch_actions)
        critic1_loss = F.mse_loss(current_Q1, target_Q)
        critic1_optimizer.zero_grad()
        critic1_loss.backward()
        critic1_optimizer.step()
    
        current_Q2 = critic2(batch_states, batch_actions)
        critic2_loss = F.mse_loss(current_Q2, target_Q)
        critic2_optimizer.zero_grad()
        critic2_loss.backward()
        critic2_optimizer.step()
         
        # Update Actor Networks
        if i_ep % parameter_update_delay == 0:
            actor_loss = -critic1(batch_states, actor(batch_states)).mean()
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()
            

        # Update the frozen target models - arbitrarily choose 100
        if i_ep % 100 == 0:
            actor_target.load_state_dict(actor.state_dict())
            critic1_target.load_state_dict(critic1.state_dict())
            critic2_target.load_state_dict(critic2.state_dict())

        # Summary stats for training:
        av_loss = critic1_loss + critic2_loss
        writer.add_scalar("loss", av_loss, i_ep)
        writer.add_scalar("Av. Q", av_Q, i_ep)
        writer.add_scalar("Max. Q", max_Q, i_ep)
    
    if timesteps_since_eval >= eval_freq:
        # evaluate
        timesteps_since_eval %= eval_freq
        avg_reward = 0.0
        col = 0
        for _ in range(eval_episodes):
            count = 0
            state = env.reset()
            done = False
            while not done and count <= max_steps:
                action = actor(np.array(state)).cpu().data.numpy().flatten()
                a_in = [(action[0] + 1) / 2, action[1]]
                state, reward, done, _ = env.step(a_in)
                avg_reward += reward
                count += 1
                if reward < -90:
                    col += 1
        avg_reward /= eval_episodes
        avg_col = col / eval_episodes
        print("..............................................")
        print(
            "Average Reward over %i Evaluation Episodes, Epoch %i: %f, %f"
            % (eval_episodes, epoch, avg_reward, avg_col)
        )
        print("..............................................")
        epoch +=1
        save(actor.state_dict(), critic1.state_dict(), critic2.state_dict(), file_name, directory="./pytorch_models")
    
    # end of episode, reset state
    state = env.reset()
    timesteps_since_eval += 1
    print("episodes:")
    print(i_ep)

#save(actor.state_dict(), critic1.state_dict(), critic2.state_dict(), file_name, directory="./models")
