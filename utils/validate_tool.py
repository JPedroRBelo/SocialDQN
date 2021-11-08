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
from pynput import keyboard
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

	max_total_fails = -3
	max_total_success = 1.5

   
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




def plot_rewards(scores,name,params,i_episode,save=False):


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

	max_total_fails = -3
	max_total_success = 1.5

   
	major_ticks = np.arange(max_total_fails, max_total_success, 0.5)
	minor_ticks = np.arange(max_total_fails, max_total_success, 0.1)

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

def compare_actions(folders,labels,action):
	params = PARAMETERS['SimDRLSR']
	neutral_reward = params['neutral_reward']
	hs_success_reward = params['hs_success_reward']
	hs_fail_reward = params['hs_fail_reward']
	eg_success_reward = params['eg_success_reward']
	eg_fail_reward = params['eg_fail_reward']
	ep_fail_reward = params['ep_fail_reward']

	results_rewards = []
	average_scores = []
	for fol,l in zip(folders,labels):
		pth = os.path.join(fol,'scores/action_reward_history.dat')
		rewards = torch.load(pth)
		results_rewards.append(rewards)
		v_hspos = []
		v_hsneg = []
		v_wvpos = []
		v_wvneg = []
		v_wave  = []
		v_wait  = []
		v_look  = []
		ep_durations = []
		scores = []  
		scores_window = deque(maxlen=100)

		for i in range(len(rewards)):
			

			hspos = 0
			hsneg = 0
			wave = 0
			wvpos = 0
			wvneg = 0
			wait = 0
			look = 0
			for step in range(len(rewards[i])):		
				if(len(rewards[i])>0 ):
					action = rewards[i][step][0]
					reward = rewards[i][step][1]

					if action == 3 :
						if reward ==  hs_success_reward:
							hspos = hspos+1
						elif reward == hs_fail_reward: 
							hsneg = hsneg+1
					
					elif action == 0 :
						wait = wait+1
					elif action == 1 :
						look = look+1
					elif action == 2 :
						
						
						wave = wave+1
						if reward == eg_success_reward:
							wvpos = wvpos+1
						elif reward == eg_fail_reward:
							wvneg = wvneg+1
					else:
						print(reward)
			
		
			
			'''
			print('###################')
			print('Epoch\t\t',i+1)
			print('Steps:\t\t',len(rewards[i]))	
			print('Wait\t\t',wait)
			print('Look\t\t',look)
			print('Wave\t\t',wave)
			print('Wv. Suc.\t',wvpos)
			print('Wv. Fail\t',wvneg)
			print('HS Suc.\t',hspos)
			print('HS Fail\t',hsneg)
			
			if(not wvpos+wvneg == 0):
				print('Wave Acuracy\t',((wvpos)/(wvpos+wvneg)))
			if(not hspos+hsneg == 0):
				print('HS Acuracy\t',((hspos)/(hspos+hsneg)))	
			'''
			ep_durations.append(len(rewards[i]))
			v_wait.append(wait)
			v_look.append(look)
			v_wave.append(wave)
			v_wvneg.append(wvneg)
			v_wvpos.append(wvpos)
			v_hspos.append(hspos)
			v_hsneg.append(hsneg)


			plot_value = len(rewards[i])
			arg = action
			#plot_value = 1	
			if (arg == 'wv'):
				plot_value = wave/plot_value
			elif (arg == 'wv positive'):
				plot_value = wvpos/plot_value
			elif (arg == 'wv negative'):
				plot_value = wvneg/plot_value
			elif (arg == 'hs'):
				plot_value = (hspos+hsneg)/plot_value
			elif (arg == 'hs positive'):
				plot_value = hspos/plot_value
			elif (arg == 'hs negative'):
				plot_value = hsneg/plot_value
			elif (arg == 'wait'):
				plot_value = wait/plot_value
			elif (arg == 'look'):
				plot_value = look/plot_value

			scores_window.append(plot_value)
			scores.append([plot_value, np.mean(scores_window), np.std(scores_window)])

		average_scores.append([calc_average_scores(scores,maxlen=100),l])

	plot_rewards(average_scores,'result',params,100)
def compare_cumulative_rewards(folders,labels):

	scores = []
	for fol,l in zip(folders,labels): 
		score = pd.read_csv(os.path.join(fol,'scores/NeuralQLearner_simDRLSR_batch_128_lr_3E-04_trained_15000_episodes.csv'), sep=',')
		scores.append([calc_average_scores(score['scores'],maxlen=100),l])

	#scores1 = pd.read_csv(os.path.join(folders[i],'scores/NeuralQLearner_simDRLSR_batch_128_lr_3E-04_trained_15000_episodes.csv'), sep=',') 
	#scores2 = pd.read_csv('results/20211103_084843/scores/NeuralQLearner_simDRLSR_batch_128_lr_3E-04_trained_15000_episodes.csv', sep=',') 
	#scores3 = pd.read_csv('results/20211107_055205/scores/NeuralQLearner_simDRLSR_batch_128_lr_3E-04_trained_15000_episodes.csv', sep=',') 
	params = PARAMETERS['SimDRLSR']
	neutral_reward = params['neutral_reward']
	hs_success_reward = params['hs_success_reward']
	hs_fail_reward = params['hs_fail_reward']
	eg_success_reward = params['eg_success_reward']
	eg_fail_reward = params['eg_fail_reward']
	ep_fail_reward = params['ep_fail_reward']
	average_scores = []
	#average_scores.append([calc_average_scores(scores1['scores'],maxlen=100),'Gray and Face States'])
	#average_scores.append([calc_average_scores(scores2['scores'],maxlen=100),'Train after 4 Steps'])
	#average_scores.append([calc_average_scores(scores3['scores'],maxlen=100),'Only Gray State'])
	
	#recalc_scores2 = calc_average_scores(scores['scores'],maxlen=500)
	#box_plot(average_scores)
	plot(scores,'result',params,100)

def main():
	folders = []
	folders.append('results/20211030_064156')
	#folders.append('results/20211103_084843')
	folders.append('results/20211107_055205')

	labels = []
	labels.append('Gray and Face States')
	#labels.append('Train after 4 Steps')
	labels.append('Only Gray State')

	while True:
		print("=========================")		
		print("|1 :Culmulative Rewards\t|")
		print("|2 :Wave\t\t|")
		print("|3 :Wave\t\t|")
		print("|4 :Wave\t\t|")
		print("|X :Exit\t\t|")
		print("=========================")
		with keyboard.Events() as events:
			# Block for as much as possible
			event = events.get(1e6)
			if event.key == keyboard.KeyCode.from_char('1'):
				compare_cumulative_rewards(folders,labels)
			elif event.key == keyboard.KeyCode.from_char('2'):
				compare_actions(folders,labels,'wv')
			elif event.key == keyboard.KeyCode.from_char('3'):
				compare_actions(folders,labels,'wv positive')
			elif event.key == keyboard.KeyCode.from_char('4'):
				compare_actions(folders,labels,'wv negative')
			elif event.key == keyboard.KeyCode.from_char('x'):
				break;			
			else:
				print('\nIncorrect key...')





if __name__ == "__main__":
	main()