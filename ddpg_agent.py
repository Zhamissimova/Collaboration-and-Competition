import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    
    def __init__(self, state_size, action_size, random_seed):
        """Initialize an Agent object.
        
        Params
        ======
            ob_size (int): dimension of one agent's observation
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.full_state_size = (state_size + action_size)*2
        self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(self.full_state_size, 1, random_seed).to(device)
        self.critic_target = Critic(self.full_state_size, 1, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(action_size, random_seed) 
        
    def act(self, states, add_noise=True):
        """Returns actions for given state as per current policy.
        :param state: observation from the single agent
        """
        state = torch.from_numpy(states).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)
    
    def reset(self):
        self.noise.reset()
        
        
    def learn(self, s0, s1, a0, a1, r0, r1, next_s0, next_s1, d0, d1, a_next0, a_next1, a_pred0, a_pred1):
        # ---------------------------- update critic ---------------------------- #
        self.critic_optimizer.zero_grad()

        # Get predicted next-state actions and Q values from target models        
        with torch.no_grad():
            Q_targets_next = self.critic_target(next_s0, next_s1, a_next0, a_next1)
        
        # Compute Q targets for current states (y_i)
        Q_targets = r0 + (GAMMA * Q_targets_next * (1 - d0))
        # Compute critic loss
        Q_expected = self.critic_local(s0, s1, a0, a1)
        
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        critic_loss.backward(retain_graph=True)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        self.actor_optimizer.zero_grad()

        # Compute actor loss
        actor_loss = -self.critic_local(s0, s1, a_pred0, a_pred1).mean()
        
        # Minimize the loss
        actor_loss.backward(retain_graph=True)
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)                    
        
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)        

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state