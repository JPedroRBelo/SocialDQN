import torch
import numpy as np


folder = "20220413_110912"
emotion = torch.load("results/"+folder+"/scores/social_signals_history.dat")

for dat in emotion:
	if(dat != None):
		for e in dat:
			print(e.cpu().detach().numpy())