import sys
sys.path.append('../')
from utils.misc import *
import math
import torch
import numpy as np
import pickle
import os
import inspect
import matplotlib.pyplot as plt
from config.hyperparams import *




def plot(save=False):

	df = pandas.read_csv('results6.csv', sep=',')

	#df = pandas.DataFrame(scores,columns=['scores','average_scores','std'])
	fig = plt.figure(num=2,figsize=(10, 5))
	plt.clf()


	ax = fig.add_subplot(111)
	print(len(df['scores']))
	episode = np.arange(len(df['scores']))
	plt.plot(episode,df['average_scores'])
	plt.fill_between(episode,df['average_scores'].add(df['std']),df['average_scores'].sub(df['std']),alpha=0.3)

	ax.legend(['Average scores'])
	plt.ylabel('Score')
	plt.xlabel('Episode')

	'''
	if(df['average_scores'].size<=1):
		max_total_fails = params['hs_fail_reward']*params['t_steps']
	else:
		max_total_fails = min(df['scores'].min(),df['average_scores'].min())
	if max_total_fails < 0:
		 max_total_fails = int(math.floor(max_total_fails))
		 max_total_success = 1.5 
	else:
		max_total_fails = int(math.ceil(max_total_fails)) 
		max_total_success = 1.1
	'''
	max_total_success = 1.5
	max_total_fails = -4
   
	major_ticks = np.arange(max_total_fails, max_total_success, 0.5)
	minor_ticks = np.arange(max_total_fails, max_total_success, 0.1)

	#ax.set_xticks(major_ticks)
	#ax.set_xticks(minor_ticks, minor=True)
	ax.set_yticks(major_ticks)
	ax.set_yticks(minor_ticks, minor=True)
	ax.axhline(1, color='gray', linewidth=0.5)
	ax.axhline(0, color='gray', linewidth=0.5)
	ax.axhline(-1, color='gray', linewidth=0.5)
	# And a corresponding grid
	#ax.grid(which='both')#,axis='y')

	# Or if you want different settings for the grids:
	ax.grid(which='minor', alpha=0.2)
	ax.grid(which='major', alpha=0.5)
	if(save):
		fig.savefig('scores/%s_%s_batch_%d_lr_%.E_trained_%d_episodes.png'% (agent.name,env_name,params['batch_size'],params['learning_rate'],i_episode))   # save the figure to file

	plt.show() # pause a bit so that plots are updated

plot()
