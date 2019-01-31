# Collaboration-and-Competition
Submission of third project in Deep Reinforcement Learning Nanodegree Program in Udacity

## Background
The goal of the project is to teach two agents to play tennis game by controlling two rackets and trying to make a ball bounce over a net. This is a continuous reinforcement learning task for 2 competing agents, hence, multi-agent actor-critic algorithm ,which is considered as extenstion of Deep Deterministic Policy Gradients (DDPG) algorithm is used to solve this Tennis environment problem.

## Environment
This “Tennis” environment is provided by The Unity Machine Learning Agents Toolkit (ML-Agents). This Unity plugin, which Machine Learning Agents (ML-Agents) is an open-source Unity plugin that enables games and simulations to serve as environments for training intelligent agents. In other words, Unity gives a chance for machine learning scientists to train their algorithms and visualize how the trained agent is performing through animations (which is really cool and fun!). 
### State space: 24. 
It includes variables corresponding to the position and velocity of the ball and racket.
### Action space: 2, continuous.
It corresponds to movement toward (or away from) the net, and jumping.
### Reward.
A reward of +0.1 is provided if an agent hits the ball over the net.
A reward of -0.1 is provided if an agent lets a ball hit the ground or hits the ball out of bounds.
### Number of agents: 2.
2 identical agents, each has its own copy of the environment, where actor networks are aware only about agent's own observations and critic network knows all actions and states of other agents.

### Task
The task is episodic, and in order to solve the environment, the average score among agents over 100 consecutive episodes must be higher than 0.5.
The problem was solved with MADDPG algorithm in 4408 episodes.
The code was written in python v3, using pytorch. GPU is provides fairly good speed up. 

### How to run:
There are 5 files required to run the project. 
* Tennis.ipynb is a Jupiter notebook with main code and the only file that need to be run.
* maddpg_agents.py is a python script which contains our maddpg reinforcement learning algorithm, which describes how ddpg agents shoul interact with each other.
* ddpg_agent.py is a python script which contains our actor-critic reinforcement learning agents.
* model.py is a python script which contains our 2 separate deep neural networks.
* "checkpoint" files with .pth format contain saved weights of actor and critic networks for all agents.

### Useful links:
* You can study DDPG paper [here](https://arxiv.org/abs/1509.02971)
* You can study MADDPG paper [here](https://arxiv.org/pdf/1706.02275)
* Tennis Environment [here](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md)
* More about Unity agents [here](https://github.com/Unity-Technologies/ml-agents)
* More about Udacity Deep Reinforcement Learning [here](https://udacity.com)
* If you got really excited about RL, then you can read book written by “father” of Reinforcement Learning [here](http://incompleteideas.net/book/the-book-2nd.html)
