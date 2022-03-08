# -*- coding: utf-8 -*-
import sys
sys.path.append('../')
import numpy as np
import random
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.optim as optim

from network.SDQNetwork import *
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# -------------------------------------------------------------------- #
# Neural Q-Learning
# -------------------------------------------------------------------- #
class MultimodalAgent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, param, seed):
        """Initialize an Agent object.
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.name = 'MultimodalNeuralQLearner'
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.device = param['device']

        # Learning params
        self.gamma = param['gamma']

        # Q-Network (Fully connected)
        #self.Q_network = QNetwork(state_size, action_size, param['hidden_layers'], seed).to(self.device)
        #self.Q_network_target = QNetwork(state_size, action_size, param['hidden_layers'], seed).to(self.device)
        self.gray_Q_network = DQN(param)
        self.gray_Q_network_target = DQN(param)
        self.depth_Q_network = DQN(param,enable_social_signs=False)
        self.depth_Q_network_target = DQN(param,enable_social_signs=False)
        self.gray_optimizer = optim.Adam(self.gray_Q_network.parameters(), lr=param['learning_rate'])
        self.depth_optimizer = optim.Adam(self.depth_Q_network.parameters(), lr=param['learning_rate'])
        #self.optimizer = optim.RMSprop(self.Q_network.parameters(), lr=param['learning_rate'])

        # Initialize update parameters
        self.t_updates = 0
        self.fix_target_updates = param['fix_target_updates']
        self.thau = param['thau']

        # Print model summary
        #print(self.Q_network)

    def greedy(self, gray_state,depth_state):
        """Returns actions for given state as per current policy.
        Params
        ======
            state (array_like): current state
        """
        #state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.gray_Q_network.eval()
        self.depth_Q_network.eval()
        with torch.no_grad():
            gray_action_values = self.gray_Q_network(gray_state)[0]
            depth_action_values = self.depth_Q_network(depth_state)[0]
        self.gray_Q_network.train()
        self.depth_Q_network.train()
        # Greedy action selection
        tg = 0
        td = 0
        for i in range(self.action_size):
            tg += gray_action_values[i]
            td += depth_action_values[i]
        #ng = gray_action_values/(gray_action_values.max()/1.0)
        #nd = depth_action_values/(depth_action_values.max()/1.0)
        q_fus=((tg)*0.5)+((td)*0.5)
        action = np.argmax(q_fus.cpu().data.numpy())
        return action


    def eGreedy(self, gray_state,depth_state, eps=0.):
        """Returns actions for given state as per current policy.
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        # Epsilon-greedy action selection
        if random.random() > eps:
            return self.greedy(gray_state,depth_state)
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        gray_states,depth_states, actions, rewards, next_gray_states, next_depth_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        gray_Q_targets_next = self.gray_Q_network_target(next_gray_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        gray_Q_targets = rewards + (self.gamma * gray_Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        gray_Q_expected = self.gray_Q_network(gray_states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(gray_Q_expected, gray_Q_targets)
        # Minimize the loss
        self.gray_optimizer.zero_grad()
        loss.backward()
        self.gray_optimizer.step()

        # Update target network
        if (self.t_updates % self.fix_target_updates) == 0:
            self.update_target(self.gray_Q_network, self.gray_Q_network_target, self.thau)

        # Get max predicted Q values (for next states) from target model
        depth_Q_targets_next = self.depth_Q_network_target(next_depth_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        depth_Q_targets = rewards + (self.gamma * depth_Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        depth_Q_expected = self.depth_Q_network(depth_states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(depth_Q_expected, depth_Q_targets)
        # Minimize the loss
        self.depth_optimizer.zero_grad()
        loss.backward()
        self.depth_optimizer.step()

        # Update target network
        if (self.t_updates % self.fix_target_updates) == 0:
            self.update_target(self.depth_Q_network, self.depth_Q_network_target, self.thau)

        self.t_updates += 1

    def update_target(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def export_network(self,filename):
        torch.save(self.gray_Q_network.state_dict(), '%sgray.pth'% (filename))
        torch.save(self.depth_Q_network.state_dict(), '%sdepth.pth'% (filename))

    def import_network(self,filename):
        self.gray_Q_network.load_state_dict(torch.load('%sgray.pth'% (filename)))
        self.depth_Q_network.load_state_dict(torch.load('%sdepth.pth'% (filename)))

# -------------------------------------------------------------------- #
# EOF
# -------------------------------------------------------------------- #