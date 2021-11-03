import sys
import os
import inspect
sys.path.append('../')
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from utils.misc import *
import math
import torch
import numpy as np
import pickle
import os

import matplotlib.pyplot as plt
from config.hyperparams import *
import pandas as pd




def plot(scores,name,params,i_episode,save=False):


	fig = plt.figure(num=2,figsize=(10, 5))
	plt.clf()


	ax = fig.add_subplot(111)
	episode = np.arange(len(scores[0][0]))
	
	average_scores = []
	count = 1
	legends = []
	for scr,legend in scores:
		df = pandas.DataFrame(scr,columns=['scores','average_scores','std'])
		plt.plot(episode,df['average_scores'])
		plt.fill_between(episode,df['average_scores'].add(df['std']),df['average_scores'].sub(df['std']),alpha=0.3)
		legends.append(legend)
		count += 1
	ax.legend(legends)
		
	#plt.fill_between(episode,df['average_scores'].add(df['std']),df['average_scores'].sub(df['std']),alpha=0.3)

	
	plt.ylabel('Score')
	plt.xlabel('Episode')


   
	#major_ticks = np.arange(max_total_fails, max_total_success, 0.5)
	#minor_ticks = np.arange(max_total_fails, max_total_success, 0.1)

	#ax.set_xticks(major_ticks)
	#ax.set_xticks(minor_ticks, minor=True)
	#ax.set_yticks(major_ticks)
	#ax.set_yticks(minor_ticks, minor=True)
	#ax.axhline(1, color='gray', linewidth=0.5)
	#ax.axhline(0, color='gray', linewidth=0.5)
	#ax.axhline(-1, color='gray', linewidth=0.5)
	# And a corresponding grid
	#ax.grid(which='both')#,axis='y')

	# Or if you want different settings for the grids:
	ax.grid(which='minor', alpha=0.2)
	ax.grid(which='major', alpha=0.5)
	if(save):
		fig.savefig('scores/%s_%s_batch_%d_lr_%.E_trained_%d_episodes.png'% (agent.name,env_name,params['batch_size'],params['learning_rate'],i_episode))   # save the figure to file

	plt.show() # pause a bit so that plots are updated


def calc_average_scores(old_scores,maxlen=100):
	scores = []								 # list containing scores from each episode
	scores_window = deque(maxlen=maxlen)   # last (window_size) scores
	for s in old_scores:
		scores_window.append(s)
		scores.append([s, np.mean(scores_window), np.std(scores_window)])
	return scores


def box_plot(results):
	fig = plt.figure()
	fig.suptitle('Algorithm Comparison')
	ax = fig.add_subplot(111)
	plt.boxplot(results)
	#ax.set_xticklabels(names)
	plt.show()


def main():

	
	scores1 = pd.read_csv('results/20211030_064156/scores/NeuralQLearner_simDRLSR_batch_128_lr_3E-04_trained_15000_episodes.csv', sep=',') 
	scores2 = pd.read_csv('results/20211103_084843/scores/NeuralQLearner_simDRLSR_batch_128_lr_3E-04_trained_15000_episodes.csv', sep=',') 
	params = PARAMETERS['SimDRLSR']
	neutral_reward = params['neutral_reward']
	hs_success_reward = params['hs_success_reward']
	hs_fail_reward = params['hs_fail_reward']
	eg_success_reward = params['eg_success_reward']
	eg_fail_reward = params['eg_fail_reward']
	ep_fail_reward = params['ep_fail_reward']
	average_scores = []
	average_scores.append([calc_average_scores(scores1['scores'],maxlen=50),'Train after each Epoch'])
	average_scores.append([calc_average_scores(scores2['scores'],maxlen=50),'Train after 4 Steps'])
	
	#recalc_scores2 = calc_average_scores(scores['scores'],maxlen=500)
	#box_plot(average_scores)
	plot(average_scores,'result',params,100)

if __name__ == "__main__":
	main()