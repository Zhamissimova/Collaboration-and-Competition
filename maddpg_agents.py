import numpy as np
import random
import copy
from collections import namedtuple, deque
import torch

from ddpg_agent import Agent

BUFFER_SIZE = int(3e5)  # replay buffer size
BATCH_SIZE = 256        # minibatch size
GAMMA = 0.99            # discount factor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MADDPG():
    
    def __init__(self, state_size, action_size, random_seed):
        """Initialize 2 Agent objects.
        
        Params
        ======
            state_size (int): dimension of one agent's observation
            action_size (int): dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size        
        # Initialize the agents
        self.ddpg_agent0 = Agent(state_size, action_size, random_seed = 0)
        self.ddpg_agent1 = Agent(state_size, action_size, random_seed = 1)
        
        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
    
    def act(self, states, rand = False):
        """Agents act with actor_local"""
        if rand == False:
            action0 = self.ddpg_agent0.act(states[0])
            action1 = self.ddpg_agent1.act(states[1])
            actions = [action0, action1]
            return actions
        if rand == True:
            actions = np.random.randn(2, 2) 
            actions = np.clip(actions, -1, 1)
            return actions
 
        
    def step(self, states, actions, rewards, next_states, dones, learn = True):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        state0 = states[0]
        state1 = states[1]
        
        action0 = actions[0]
        action1 = actions[1]
        
        reward0 = rewards[0]
        reward1 = rewards[1]
        
        next_state0 = next_states[0]
        next_state1 = next_states[1]
        
        done0 = dones[0]
        done1 = dones[1]

        self.memory.add(state0, state1, action0, action1, reward0, reward1, next_state0, next_state1, done0, done1)
        
        if learn == True and len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)
     
    def learn(self, experiences, GAMMA):
        s0, s1, a0, a1, r0, r1, next_s0, next_s1, d0, d1 = experiences
        
        # next actions (for CRITIC network)
        a_next0 = self.ddpg_agent0.actor_target(next_s0)
        a_next1 = self.ddpg_agent1.actor_target(next_s1)
        
        # action predictions (for ACTOR network)
        a_pred0 = self.ddpg_agent0.actor_local(s0)
        a_pred1 = self.ddpg_agent1.actor_local(s1)
        
        # ddpg agents learn separately, each agent learns from its perspective, that is why states, actions, etc are swapped
        self.ddpg_agent0.learn(s0, s1, a0, a1, r0, r1, next_s0, next_s1, d0, d1, a_next0, a_next1, a_pred0, a_pred1)
        self.ddpg_agent1.learn(s1, s0, a1, a0, r1, r0, next_s1, next_s0, d1, d0, a_next1, a_next0, a_pred1, a_pred0)
    

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state0","state1", 
                                                                "action0", "action1",
                                                                "reward0", "reward1",
                                                                "next_state0", "next_state1",
                                                                "done0", "done1"])
        self.seed = random.seed(seed)
    
    def add(self, state0, state1, action0, action1, reward0, reward1, next_state0, next_state1, done0, done1):
        """Add a new experience to memory."""
        e = self.experience(state0, state1, action0, action1, reward0, reward1, next_state0, next_state1, done0, done1)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states0 = torch.from_numpy(np.vstack([e.state0 for e in experiences if e is not None])).float().to(device)
        states1 = torch.from_numpy(np.vstack([e.state1 for e in experiences if e is not None])).float().to(device)
        
        actions0 = torch.from_numpy(np.vstack([e.action0 for e in experiences if e is not None])).float().to(device)
        actions1 = torch.from_numpy(np.vstack([e.action1 for e in experiences if e is not None])).float().to(device)
        
        rewards0 = torch.from_numpy(np.vstack([e.reward0 for e in experiences if e is not None])).float().to(device)
        rewards1 = torch.from_numpy(np.vstack([e.reward1 for e in experiences if e is not None])).float().to(device)
        
        next_states0 = torch.from_numpy(np.vstack([e.next_state0 for e in experiences if e is not None])).float().to(device)
        next_states1 = torch.from_numpy(np.vstack([e.next_state1 for e in experiences if e is not None])).float().to(device)
        
        dones0 = torch.from_numpy(np.vstack([e.done0 for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        dones1 = torch.from_numpy(np.vstack([e.done1 for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states0, states1, actions0, actions1, rewards0, rewards1, next_states0, next_states1, dones0, dones1)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)        
        
        
        
