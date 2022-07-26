# -*- coding: utf-8 -*-
import sys
sys.path.append('../')
import numpy as np
import random
from collections import namedtuple, deque

import torch
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------------------------- #
# Experience replay
# -------------------------------------------------------------------- #
class MultimodalReplayBuffer():
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed,device):
        """Initialize a ReplayBuffer object.
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["gray_state","depth_state", "action", "reward", "next_gray_state","next_depth_state", "done"])
        self.seed = random.seed(seed)
        self.device = device

    def push(self, gray_state,depth_state, action, reward, next_gray_state,next_depth_state, done):
        """Add a new experience to memory."""
        e = self.experience(gray_state,depth_state, action, reward, next_gray_state,next_depth_state, done)
        self.memory.append(e)

    def recall(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        gray_images = torch.from_numpy(np.vstack([e.gray_state[0] for e in experiences if e is not None])).float().to(self.device)
        gray_social_signals = torch.from_numpy(np.vstack([e.gray_state[1] for e in experiences if e is not None])).float().to(self.device)
        depth_images = torch.from_numpy(np.vstack([e.depth_state[0] for e in experiences if e is not None])).float().to(self.device)
        depth_social_signals = torch.from_numpy(np.vstack([e.depth_state[1] for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_gray_images = torch.from_numpy(np.vstack([e.next_gray_state[0] for e in experiences if e is not None])).float().to(self.device)
        next_gray_social_signals = torch.from_numpy(np.vstack([e.next_gray_state[1] for e in experiences if e is not None])).float().to(self.device)
        next_depth_images = torch.from_numpy(np.vstack([e.next_depth_state[0] for e in experiences if e is not None])).float().to(self.device)
        next_depth_social_signals = torch.from_numpy(np.vstack([e.next_depth_state[1] for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
        gray_states = [gray_images,gray_social_signals]
        depth_states = [depth_images,depth_social_signals]
        next_gray_states = [next_gray_images,next_gray_social_signals]
        next_depth_states = [next_depth_images,next_depth_social_signals]
        #return (states, actions, rewards, next_states, dones)

        return (gray_states, depth_states, actions, rewards, next_gray_states, next_depth_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class ReplayBuffer():
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed,device):
        """Initialize a ReplayBuffer object.
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.device = device

    def push(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward,next_state, done)
        self.memory.append(e)

    def recall(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        images = torch.from_numpy(np.vstack([e.state[0] for e in experiences if e is not None])).float().to(self.device)
        emotions = torch.from_numpy(np.vstack([e.state[1] for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_images = torch.from_numpy(np.vstack([e.next_state[0] for e in experiences if e is not None])).float().to(self.device)
        next_emotions = torch.from_numpy(np.vstack([e.next_state[1] for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
        states = [images,emotions]
        next_states = [next_images,next_emotions]
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

# -------------------------------------------------------------------- #
# EOF
# -------------------------------------------------------------------- #