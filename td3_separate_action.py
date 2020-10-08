import os
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from replay_buffer import ReplayBuffer

def evaluate(network, eval_episodes=10):
  avg_reward = 0.
  for _ in range(eval_episodes):
    state = env.reset()
    done = False
    while not done:
      action = network.get_action(np.array(state))
      state, reward, done, _ = env.step(action)
      avg_reward += reward
  avg_reward /= eval_episodes
  print ("..............................................")
  print ("Average Reward over %i Evaluation Episodes: %f"  % (eval_episodes, avg_reward))
  print ("..............................................")
  return avg_reward


class Actor(nn.Module):

  def __init__(self, state_dim, max_action):
    super(Actor, self).__init__()

    self.layer_1 = nn.Linear(state_dim, 400)
    self.layer_2 = nn.Linear(400, 300)
    # output 3 joint actions of a leg
    self.layer_3 = nn.Linear(300, 3)
    # output other 3 joint actions for the other leg based on the output from the first leg
    self.layer_4 = nn.Linear(303, 3)
    self.max_action = max_action
    self.soft = nn.Softsign()

  def forward(self, s):
    s =F.relu(self.layer_1(s))
    s =F.relu(self.layer_2(s))
    t1 = self.soft(self.layer_3(s))
    t2 = self.soft(self.layer_4(torch.cat([s,t1],axis = 1)))
    # combine the actions for final output
    a = self.max_action * torch.cat([t2,t1], axis = 1)
    return a


class Critic(nn.Module):

  def __init__(self, state_dim, action_dim):
    super(Critic, self).__init__()
    #first critic
    self.layer_1 = nn.Linear(state_dim, 400)
    # temporary layer for state
    self.layer_2_s = nn.Linear(400, 300)
    # temporary layer for action
    self.layer_2_a = nn.Linear(action_dim, 300)
    self.layer_3 = nn.Linear(300, 1)

    #second critic
    self.layer_4 = nn.Linear(state_dim, 400)
    # temporary layer for state
    self.layer_5_s = nn.Linear(400, 300)
    # temporary layer for action
    self.layer_5_a = nn.Linear(action_dim, 300)
    self.layer_6 = nn.Linear(300, 1)


  def forward(self, s, a):
    
    #first critic
    s1 =F.relu(self.layer_1(s))
    t1 = self.layer_2_s(s1)
    t2 = self.layer_2_a(a)
    # combine the state and action temporary layers in the final dense layer
    s1 = F.relu(t1+t2)
    q1 = self.layer_3(s1)

    #second critic
    s2 =F.relu(self.layer_4(s))
    t3 = self.layer_5_s(s2)
    t4 = self.layer_5_a(a)
    # combine the state and action temporary layers in the final dense layer
    s2 = F.relu(t3+t4)
    q2 = self.layer_6(s2)
    return q1, q2


#TD3 network
class TD3(object):

  def __init__(self, state_dim, action_dim, max_action):
    # Initialize the Actor network
    self.actor = Actor(state_dim, max_action).to(device)
    self.actor_target = Actor(state_dim, max_action).to(device)
    self.actor_target.load_state_dict(self.actor.state_dict())
    self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

    # Initialize the Critic networks
    self.critic = Critic(state_dim, action_dim).to(device)
    self.critic_target = Critic(state_dim, action_dim).to(device)
    self.critic_target.load_state_dict(self.critic.state_dict())
    self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

    self.max_action = max_action

  # Function to get the action from the actor
  def get_action(self, state):
    state = torch.Tensor(state.reshape(1,-1)).to(device)
    return self.actor(state).cpu().data.numpy().flatten()


  # training cycle
  def train(self, replay_buffer, iterations, batch_size=100, discount = 0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
    for it in range(iterations):
      # sample a batch from the replay buffer
      batch_states, batch_actions, batch_rewards, batch_dones, batch_next_states = replay_buffer.sample_batch(batch_size)
      state = torch.Tensor(batch_states).to(device)
      next_state = torch.Tensor(batch_next_states).to(device)
      action = torch.Tensor(batch_actions).to(device)
      reward = torch.Tensor(batch_rewards).to(device)
      done = torch.Tensor(batch_dones).to(device)

      # Obtain the estimated action from the next state by using the actor-target
      next_action = self.actor_target(next_state)

      # Add noise to the action
      noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(device)
      noise = noise.clamp(-noise_clip,noise_clip)
      next_action= (next_action+noise).clamp(-self.max_action,self.max_action)

      # Calculate the Q values from the critic-target network for the next state-action pair
      target_Q1, target_Q2 = self.critic_target(next_state, next_action)

      # Select the minimal Q value from the 2 calculated values
      target_Q = torch.min(target_Q1, target_Q2)

      # Calculate the final Q value from the target network parameters by using Bellman equation
      target_Q = reward + ((1-done)*discount*target_Q).detach()

      # Get the Q values of the basis networks with the current parameters
      current_Q1, current_Q2 = self.critic(state, action)

      # Calculate the loss between the current Q value and the target Q value
      loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

      # Perform the gradient descent
      self.critic_optimizer.zero_grad()
      loss.backward()
      self.critic_optimizer.step()

      if it%policy_freq==0:
        # Maximize the the actor output value by performing gradient descent on negative Q values (essentially perform gradient ascent)
        actor_grad, _ = self.critic(state, self.actor(state))
        actor_grad = -actor_grad.mean()
        self.actor_optimizer.zero_grad()
        actor_grad.backward()
        self.actor_optimizer.step()

        # Use soft update to update the actor-target network parameters by infusing small amount of current parameters
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
          target_param.data.copy_(tau*param.data +(1-tau)*target_param.data)
        # Use soft update to update the critic-target network parameters by infusing small amount of current parameters
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
          target_param.data.copy_(tau*param.data +(1-tau)*target_param.data)


  def save(self, filename, directory):
    torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
    torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))

  def load(self, filename, directory):
    self.actor.load_state_dict(torch.load( '%s/%s_actor.pth' % (directory, filename)))
    self.critic.load_state_dict(torch.load( '%s/%s_critic.pth' % (directory, filename)))






# Set the parameters for the implementation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # cuda or cpu
env_name = "HalfCheetahBulletEnv-v0" # Name of the PyBullet environment. The network is updated for HalfCheetahBulletEnv-v0
seed = 0 # Random seed number
eval_freq = 5e3 # After how many steps to perform the evaluation
eval_ep = 10 # number of episodes for evaluation
max_timesteps = 5e5 # Maximum number of steps to perform
save_models = False # Weather to save the model or not
expl_noise = 1 # Initial exploration noise starting value in range [expl_min ... 1]
expl_decay_steps = 50000 # Number of steps over which the initial exploration noise will decay over
expl_min = 0.1 # Exploration noise after the decay in range [0...expl_noise]
batch_size = 100 # Size of the mini-batch
discount = 0.99 # Discount factor to calculate the discounted future reward (should be close to 1)
tau = 0.005 # Soft target update variable (should be close to 0)
policy_noise = 0.2 # Added noise for exploration
noise_clip = 0.5 # Maximum clamping values of the noise
policy_freq = 2 # Frequency of Actor network updates
buffer_size = 1e6 # Maximum size of the buffer
file_name = "TD3_Cheetah" #name of the file to store the policy


# Create the network storage folders
if not os.path.exists("./results"):
  os.makedirs("./results")
if save_models and not os.path.exists("./pytorch_models"):
  os.makedirs("./pytorch_models")


# Create the training environment
env = gym.make(env_name)
env.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

# Create the network
network = TD3(state_dim, action_dim, max_action)
# Create a replay buffer
replay_buffer = ReplayBuffer(buffer_size, seed)

# Create evaluation data store
evaluations = [evaluate(network,eval_ep)]


timestep = 0
timesteps_since_eval = 0
episode_num = 0
done = True

# Begin the training loop
while timestep < max_timesteps:
  
  # On termination of episode
  if done:

    if timestep != 0:
      print("Timestep: {} Episode: {} Reward: {}".format(timestep, episode_num, episode_reward))
      # Train the network on experiences
      network.train(replay_buffer, episode_timesteps, batch_size, discount, tau, policy_noise, noise_clip, policy_freq)

    # Evaluate the episode
    if timesteps_since_eval >= eval_freq:
      timesteps_since_eval %= eval_freq
      evaluations.append(evaluate(network,eval_ep))
      network.save(file_name, directory="./pytorch_models")
      np.save("./results/%s" % (file_name), evaluations)
    
    # Reset the environment to start a new episode
    state = env.reset()
    
    # Reset done value
    done = False
    
    # Reset the counters
    episode_reward = 0
    episode_timesteps = 0
    episode_num += 1

  # Slightly reduce the noise at every timestep
  if expl_noise > expl_min:
    expl_noise = expl_noise-((1-expl_min)/expl_decay_steps)

  # Get the ection to perform in the environment
  action = network.get_action(np.array(state))
  # Add some noise to the action for exploration
  action = (action + np.random.normal(0, expl_noise, size=env.action_space.shape[0])).clip(env.action_space.low, env.action_space.high)
  
  # Perfom the action in the environment and recieve the reward as well as the next state
  next_state, reward, done, _ = env.step(action)
  
  # Check if episode is finished
  done_bool = 0 if episode_timesteps + 1 == env._max_episode_steps else float(done)
  
  # Add the current state-action pair reward to the cumulative episode reward
  episode_reward += reward
  
  # Store the new tuple in the replay buffer
  replay_buffer.add(state, action, reward, done_bool, next_state)

  # Update the counters
  state = next_state
  episode_timesteps += 1
  timestep += 1
  timesteps_since_eval += 1

# After the training is done, evaluate the network and save it
evaluations.append(evaluate(network,eval_ep))
if save_models: network.save("%s" % (file_name), directory="./models")
np.save("./results/%s" % (file_name), evaluations)

