# -*- coding: utf-8 -*-
import sys
sys.path.append('../')
import torch
import torch.nn as nn
import torch.nn.functional as F
import config.config as cfg


class SSDQN(nn.Module):
	def __init__(self,params):
		super(SSDQN, self).__init__()
		self.noutputs=params['action_size']
		self.nfeats=params['state_size']
		self.nstates=params['nstates']
		self.kernels=params['kernels']
		self.strides=params['strides']
		self.poolsize=params['poolsize']
		self.social_state_size = params['social_state_size']
		self.nstates_social = params['nstates_social']
		self.features = nn.Sequential(
			nn.Conv2d(in_channels=self.nfeats,out_channels=self.nstates[0], kernel_size=self.kernels[0],stride=self.strides[0],padding=1),
			nn.BatchNorm2d(self.nstates[0]),
			nn.ReLU(),
			nn.MaxPool2d(self.poolsize),
			nn.Conv2d(in_channels=self.nstates[0],out_channels=self.nstates[1], kernel_size=self.kernels[1],stride=self.strides[1]),
			nn.BatchNorm2d(self.nstates[1]),
			nn.ReLU(),
			nn.MaxPool2d(self.poolsize),
			nn.Conv2d(in_channels=self.nstates[1],out_channels=self.nstates[2], kernel_size=self.kernels[1],stride=self.strides[1]),
			nn.BatchNorm2d(self.nstates[2]),
			nn.ReLU(),
			nn.MaxPool2d(self.poolsize),	
			)
		self.linear = nn.Sequential(
            nn.Linear(self.social_state_size, self.nstates_social[0]),
            nn.ReLU()
        )

		self.classifier = nn.Sequential(
			nn.Linear((self.nstates[2]*self.kernels[1]*self.kernels[1])+self.nstates_social[0],self.nstates[3]),
			nn.ReLU(),
			nn.Linear(self.nstates[3], self.noutputs),
		)



	def forward(self, state):
		conv = self.features(state[0])
		conv = conv.view(conv.size(0),self.nstates[2]*self.kernels[1]*self.kernels[1])

		ac = self.linear(state[1])
		ac = ac.view(1,self.nstates_social[0])
		print(conv.shape)
		print(ac.shape)
		x = torch.cat((conv, ac), 1)

		x = self.classifier(x)
		return x


