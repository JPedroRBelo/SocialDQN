# -*- coding: utf-8 -*-
import sys
sys.path.append('../')
import torch
import torch.nn as nn
import torch.nn.functional as F
import config.config as cfg
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class DQN(nn.Module):
	def __init__(self,params,enable_social_signs=None):
		super(DQN, self).__init__()
		self.noutputs=params['action_size']
		self.nfeats=params['state_size']
		self.nstates=params['nstates']
		self.kernels=params['kernels']
		self.strides=params['strides']
		self.poolsize=params['poolsize']
		if enable_social_signs==None:
			self.enable_social_signs=params['enable_social_signs']
		else:
			self.enable_social_signs=enable_social_signs
		self.nstates_social = params['nstates_social']
		self.social_state_size = params['social_state_size']

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
			nn.Linear(self.social_state_size, 256),
			nn.ReLU()
		)

		self.fc1 = nn.Sequential(
			nn.Linear(self.nstates[2]*self.kernels[1]*self.kernels[1],self.nstates[3]),
			nn.ReLU()
		)
		self.fc2 = nn.Sequential(
			nn.Linear(512,128),
			nn.ReLU()
		)
		self.classifier = nn.Sequential(
			nn.Linear(128, self.noutputs),
		)
		self.image_classifier = nn.Sequential(
			nn.Linear(self.nstates[2]*self.kernels[1]*self.kernels[1],self.nstates[3]),
			nn.ReLU(),
			nn.Linear(self.nstates[3], self.noutputs),
		)

	def forward(self, x):
		conv = self.features(x[0])
		conv = conv.view(conv.size(0),self.nstates[2]*self.kernels[1]*self.kernels[1])
		if(self.enable_social_signs):
			conv = self.fc1(conv)
			ac = self.linear(x[1])
			ac = ac.view(ac.size(0),self.nstates_social[0])
			cat = torch.cat((conv, ac), 1)
			cat = self.fc2(cat)
			cat = self.classifier(cat)
		else:
			cat = self.image_classifier(conv)

		return cat
