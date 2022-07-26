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

average_scores = []
wait_average_scores = []
look_average_scores = []
wave_average_scores = []
wave_positive_average_scores = []
wave_negative_average_scores = []
handshake_average_scores = []
handshake_positive_average_scores = []
handshake_negative_average_scores = []
handshake_ratio_positive_average_scores = []
handshake_ratio_negative_average_scores = []
wave_ratio_positive_average_scores = []
wave_ratio_negative_average_scores = []

MAX_EPS = 15000


def plot(scores,params,i_episode,save=False,title='',metric='Score',save_location=''):


	fig = plt.figure(num=2,figsize=(10, 5))
	plt.clf()


	ax = fig.add_subplot(111)
	
	
	average_scores = []
	count = 1
	legends = []
	maxlen = math.inf
	for scr,legend in scores:
		df = pandas.DataFrame(scr,columns=['scores','average_scores','std'])
		maxlen = min(maxlen,len(df['scores']))
	episode = np.arange(maxlen)
	for scr,legend in scores:
		df = pandas.DataFrame(scr,columns=['scores','average_scores','std'])
		plt.plot(episode,df['average_scores'][:maxlen])
		plt.fill_between(episode,df['average_scores'][:maxlen].add(df['std'][:maxlen]),df['average_scores'][:maxlen].sub(df['std'][:maxlen]),alpha=0.3)
		legends.append(legend)
		count += 1
	ax.legend(legends,loc='upper right')
		
	#plt.fill_between(episode,df['average_scores'].add(df['std']),df['average_scores'].sub(df['std']),alpha=0.3)

	plt.title(title)
	plt.ylabel(metric)
	plt.xlabel('Episode')

	max_total_fails = -3
	max_total_success = 1.5

   
	major_ticks = np.arange(0, maxlen, 0.5)
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
	#ax.set_xticks(major_ticks, minor=False)
	if(save):
		fig.savefig(os.path.join(save_location,(title+'.png')))#, format="svg")   # save the figure to file
	else:
		plt.show() # pause a bit so that plots are updated




def plot_rewards(scores,name,params,i_episode,save=False):


	fig = plt.figure(num=2,figsize=(10, 5))
	plt.clf()

	plt.legend(loc='lower right')
	ax = fig.add_subplot(111)
	episode = np.arange(len(scores[0][0])+1)
	
	average_scores = []
	count = 1
	legends = []
	maxlen = math.inf
	for scr,legend in scores:
		df = pandas.DataFrame(scr,columns=['scores','average_scores','std'])
		maxlen = min(maxlen,len(df['scores']))
	for scr,legend in scores:
		df = pandas.DataFrame(scr,columns=['scores','average_scores','std'])
		plt.plot(episode,df['average_scores'][:maxlen])
		plt.fill_between(episode,df['average_scores'][:maxlen].add(df['std'][:maxlen]),df['average_scores'][:maxlen].sub(df['std'][:maxlen]),alpha=0.3)
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



def plot_culmulative(scores,name,params,i_episode,save=False,save_location=''):


	fig = plt.figure(num=2,figsize=(10, 5))
	plt.clf()


	ax = fig.add_subplot(111)
	#episode = np.arange(len(scores[0][0]))
	plt.title('Cumulative Rewards')
	average_scores = []
	count = 1
	legends = []
	maxlen = math.inf
	for scr,legend in scores:
		df = pandas.DataFrame(scr,columns=['scores','average_scores','std'])
		maxlen = min(maxlen,len(df['scores']))
	print(maxlen)
	episode = np.arange(maxlen)
	for scr,legend in scores:
		df = pandas.DataFrame(scr,columns=['scores','average_scores','std'])
		plt.plot(episode,df['average_scores'][:maxlen])
		plt.fill_between(episode,df['average_scores'][:maxlen].add(df['std'][:maxlen]),df['average_scores'][:maxlen].sub(df['std'][:maxlen]),alpha=0.3)
		legends.append(legend)
		count += 1
	ax.legend(legends,loc='lower right')
		
	#plt.fill_between(episode,df['average_scores'].add(df['std']),df['average_scores'].sub(df['std']),alpha=0.3)

	
	plt.ylabel('Score')
	plt.xlabel('Episode')


	max_total_fails = -3
	max_total_success = 1.5

   
	
	major_ticks = np.arange(max_total_fails, max_total_success, 0.5)
	minor_ticks = np.arange(max_total_fails, max_total_success, 0.1)
	#plt.setp(ax.get_xticklabels(), rotation=15, horizontalalignment='right')

	#ax.set_xticks(major_ticks)
	#ax.set_xticks(minor_ticks, minor=True)
	ax.set_yticks(major_ticks)
	ax.set_yticks(minor_ticks, minor=True)
	major_ticks = np.arange(0, maxlen+1, (maxlen/10))
	ax.set_xticks(major_ticks, minor=False)
	ax.axhline(1, color='gray', linewidth=0.5)
	ax.axhline(0, color='gray', linewidth=0.5)
	ax.axhline(-1, color='gray', linewidth=0.5)
	# And a corresponding grid
	#ax.grid(which='both')#,axis='y')

	# Or if you want different settings for the grids:
	ax.grid(which='minor', alpha=0.2)
	ax.grid(which='major', alpha=0.5)
	if(save):
		fig.savefig(os.path.join(save_location,('culmulative_rewards.png')))   # save the figure to file
	else:
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




def calc_all_scores(folders,labels):
	print('\rLoading...', end="")
	params = PARAMETERS['SimDRLSR']
	neutral_reward = params['neutral_reward']
	hs_success_reward = params['hs_success_reward']
	hs_fail_reward = params['hs_fail_reward']
	eg_success_reward = params['eg_success_reward']
	eg_fail_reward = params['eg_fail_reward']
	ep_fail_reward = params['ep_fail_reward']
	results_dict = {}
	results_rewards = []


	for fol,l in zip(folders,labels):
		pth = os.path.join(fol,'scores','action_reward_history.dat')
		emotion_pth = os.path.join(fol,'scores','social_signals_history.dat')
		rewards = torch.load(pth)
		emotions = torch.load(emotion_pth)
		results_rewards.append(rewards)
		v_handshake = []
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

		wait_total_scores = []  
		wait_total_scores_window = deque(maxlen=100)

		look_total_scores = []  
		look_total_scores_window = deque(maxlen=100)

		wave_total_scores = []  
		wave_total_scores_window = deque(maxlen=100)

		wave_positive_scores = []  
		wave_positive_scores_window = deque(maxlen=100)

		wave_negative_scores = []  
		wave_negative_scores_window = deque(maxlen=100)

		handshake_total_scores = []  
		handshake_total_scores_window = deque(maxlen=100)

		handshake_ratio_positive_scores = []  
		handshake_ratio_positive_scores_window = deque(maxlen=100)

		handshake_ratio_negative_scores = []  
		handshake_ratio_negative_scores_window = deque(maxlen=100)

		handshake_positive_scores = []  
		handshake_positive_scores_window = deque(maxlen=100)

		handshake_negative_scores = []  
		handshake_negative_scores_window = deque(maxlen=100)

		wave_ratio_negative_scores = []  
		wave_ratio_negative_scores_window = deque(maxlen=100)


		wave_ratio_positive_scores = []  
		wave_ratio_positive_scores_window = deque(maxlen=100)

		for i in range(len(rewards)):
			hspos = 0
			hsneg = 0
			handshake = 0
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
						handshake = handshake=1
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
			
		
		
			ep_durations.append(len(rewards[i]))
			v_wait.append(wait)
			v_look.append(look)
			v_wave.append(wave)
			v_wvneg.append(wvneg)
			v_wvpos.append(wvpos)
			v_hspos.append(hspos)
			v_hsneg.append(hsneg)
			v_handshake.append(handshake)

			len_ep = len(rewards[i])
			
			#wait
			plot_value = wait/len_ep
			wait_total_scores_window.append(plot_value) 
			wait_total_scores.append([plot_value, np.mean(wait_total_scores_window), np.std(wait_total_scores_window)])

			#look
			plot_value = look/len_ep
			look_total_scores_window.append(plot_value) 
			look_total_scores.append([plot_value, np.mean(look_total_scores_window), np.std(look_total_scores_window)])

			#wave
			plot_value = wave/len_ep
			wave_total_scores_window.append(plot_value) 
			wave_total_scores.append([plot_value, np.mean(wave_total_scores_window), np.std(wave_total_scores_window)])

			#wave negative
			plot_value = wvneg/len_ep
			wave_negative_scores_window.append(plot_value) 
			wave_negative_scores.append([plot_value, np.mean(wave_negative_scores_window), np.std(wave_negative_scores_window)])

			#wave positive
			plot_value = wvpos/len_ep
			wave_positive_scores_window.append(plot_value) 
			wave_positive_scores.append([plot_value, np.mean(wave_positive_scores_window), np.std(wave_positive_scores_window)])

			#handshake
			plot_value = handshake/len_ep
			handshake_total_scores_window.append(plot_value) 
			handshake_total_scores.append([plot_value, np.mean(handshake_total_scores_window), np.std(handshake_total_scores_window)])


			#handshake positive
			plot_value = hspos/len_ep
			handshake_positive_scores_window.append(plot_value) 
			handshake_positive_scores.append([plot_value, np.mean(handshake_positive_scores_window), np.std(handshake_positive_scores_window)])

			#handshake negative
			plot_value = hsneg/len_ep
			handshake_negative_scores_window.append(plot_value) 
			handshake_negative_scores.append([plot_value, np.mean(handshake_negative_scores_window), np.std(handshake_negative_scores_window)])

			#handshake negative ratio
			if(hsneg+hspos==0):
				neg_plot_value = 0
				pos_plot_value = 0
			else:
				neg_plot_value = hsneg/(hsneg+hspos)
				pos_plot_value = hspos/(hsneg+hspos)

			handshake_ratio_negative_scores_window.append(neg_plot_value) 
			handshake_ratio_negative_scores.append([neg_plot_value, np.mean(handshake_ratio_negative_scores_window), np.std(handshake_ratio_negative_scores_window)])

			#handshake positive ratio
			handshake_ratio_positive_scores_window.append(pos_plot_value) 
			handshake_ratio_positive_scores.append([pos_plot_value, np.mean(handshake_ratio_positive_scores_window), np.std(handshake_ratio_positive_scores_window)])

			if(wvneg+wvpos==0):
				neg_plot_value = 0
				pos_plot_value = 0
			else:
				neg_plot_value = wvneg/(wvneg+wvpos)
				pos_plot_value = wvpos/(wvneg+wvpos)

			wave_ratio_negative_scores_window.append(neg_plot_value) 
			wave_ratio_negative_scores.append([neg_plot_value, np.mean(wave_ratio_negative_scores_window), np.std(wave_ratio_negative_scores_window)])

			#handshake positive ratio
			wave_ratio_positive_scores_window.append(pos_plot_value) 
			wave_ratio_positive_scores.append([pos_plot_value, np.mean(wave_ratio_positive_scores_window), np.std(wave_ratio_positive_scores_window)])
			

			#plot_value = len(rewards[i])
			#scores_window.append(plot_value)
			#scores.append([plot_value, np.mean(scores_window), np.std(scores_window)])

		print('\rCalculating...', end="")
		#average_scores.append([calc_average_scores(scores,maxlen=100),l])
		wait_average_scores.append([calc_average_scores(wait_total_scores,maxlen=100),l])
		look_average_scores.append([calc_average_scores(look_total_scores,maxlen=100),l])
		wave_average_scores.append([calc_average_scores(wave_total_scores,maxlen=100),l])
		wave_positive_average_scores.append([calc_average_scores(wave_positive_scores,maxlen=100),l])
		wave_negative_average_scores.append([calc_average_scores(wave_negative_scores,maxlen=100),l])
		handshake_average_scores.append([calc_average_scores(handshake_total_scores,maxlen=100),l])
		handshake_positive_average_scores.append([calc_average_scores(handshake_positive_scores,maxlen=100),l])
		handshake_negative_average_scores.append([calc_average_scores(handshake_negative_scores,maxlen=100),l])
		handshake_ratio_positive_average_scores.append([calc_average_scores(handshake_ratio_positive_scores,maxlen=100),l])
		handshake_ratio_negative_average_scores.append([calc_average_scores(handshake_ratio_negative_scores,maxlen=100),l])
		wave_ratio_positive_average_scores.append([calc_average_scores(wave_ratio_positive_scores,maxlen=100),l])
		wave_ratio_negative_average_scores.append([calc_average_scores(wave_ratio_negative_scores,maxlen=100),l])

	print('\rLoaded!\t', end="")
	print('\n')

def compare_cumulative_rewards(folders,labels,save=False,save_location=''):

	scores = []
	for fol,l in zip(folders,labels): 
		files = os.path.join(fol,'scores')
		folder_content = os.listdir(files)
		#filename = 'NeuralQLearner_simDRLSR_batch_128_lr_3E-04_trained'
		#filename = 'NeuralQLearner_simDRLSR_batch'
		filename = 'MultimodalNeuralQLearner_simDRLSR'		
		candidates = find_files_candidates(filename,folder_content)
		if(len(candidates)==0):
			filename = 'NeuralQLearner_simDRLSR'		
			candidates = find_files_candidates(filename,folder_content)
		print(candidates)
		#candidates = [path for path in folder_content if (path.startswith(filename)and path.endswith('.csv'))]
		score = pd.read_csv(os.path.join(fol,'scores',candidates[0]), sep=',',nrows=MAX_EPS)
		scores.append([calc_average_scores(score['scores'],maxlen=250),l])

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
	plot_culmulative(scores,'result',params,100,save=save,save_location=save_location)


def calc_with_emotions(folders,labels):
	print('\rLoading...', end="")
	params = PARAMETERS['SimDRLSR']
	neutral_reward = params['neutral_reward']
	hs_success_reward = params['hs_success_reward']
	hs_fail_reward = params['hs_fail_reward']
	eg_success_reward = params['eg_success_reward']
	eg_fail_reward = params['eg_fail_reward']
	ep_fail_reward = params['ep_fail_reward']
	results_dict = {}
	results_rewards = []


	for fol,l in zip(folders,labels):
		pth = os.path.join(fol,'scores','action_reward_history.dat')
		emotion_pth = os.path.join(fol,'scores','social_signals_history.dat')
		rewards = torch.load(pth)
		emotions = torch.load(emotion_pth)
		results_rewards.append(rewards)
		v_handshake = []
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

		wait_total_scores = []  
		wait_total_scores_window = deque(maxlen=100)

		look_total_scores = []  
		look_total_scores_window = deque(maxlen=100)

		wave_total_scores = []  
		wave_total_scores_window = deque(maxlen=100)

		wave_positive_scores = []  
		wave_positive_scores_window = deque(maxlen=100)

		wave_negative_scores = []  
		wave_negative_scores_window = deque(maxlen=100)

		handshake_total_scores = []  
		handshake_total_scores_window = deque(maxlen=100)

		handshake_ratio_positive_scores = []  
		handshake_ratio_positive_scores_window = deque(maxlen=100)

		handshake_ratio_negative_scores = []  
		handshake_ratio_negative_scores_window = deque(maxlen=100)

		handshake_positive_scores = []  
		handshake_positive_scores_window = deque(maxlen=100)

		handshake_negative_scores = []  
		handshake_negative_scores_window = deque(maxlen=100)

		wave_ratio_negative_scores = []  
		wave_ratio_negative_scores_window = deque(maxlen=100)


		wave_ratio_positive_scores = []  
		wave_ratio_positive_scores_window = deque(maxlen=100)



		for i in range(len(rewards)):
			hspos = 0
			hsneg = 0
			handshake = 0
			wave = 0
			wvpos = 0
			wvneg = 0
			wait = 0
			look = 0

			emotion_ep = 1
			for step in range(len(rewards[i])):		
				if(len(rewards[i])>0 ):
					emotion = emotions[i][step].numpy()
					n_emotion = convert_one_hot_to_number(emotion[0])
					if(n_emotion>0):
						emotion_ep = n_emotion
						break;
			if(str(emotion_ep) == l):
				for step in range(len(rewards[i])):		
					if(len(rewards[i])>0 ):
						action = rewards[i][step][0]
						reward = rewards[i][step][1]
						#emotion = emotions[i][step].numpy()
						#n_emotion = convert_one_hot_to_number(emotion[0])
						#print(n_emotion)

						if action == 3 :
							handshake = handshake=1
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
				
			
			
				ep_durations.append(len(rewards[i]))
				v_wait.append(wait)
				v_look.append(look)
				v_wave.append(wave)
				v_wvneg.append(wvneg)
				v_wvpos.append(wvpos)
				v_hspos.append(hspos)
				v_hsneg.append(hsneg)
				v_handshake.append(handshake)

				len_ep = len(rewards[i])
				
				#wait
				plot_value = wait/len_ep
				wait_total_scores_window.append(plot_value) 
				wait_total_scores.append([plot_value, np.mean(wait_total_scores_window), np.std(wait_total_scores_window)])

				#look
				plot_value = look/len_ep
				look_total_scores_window.append(plot_value) 
				look_total_scores.append([plot_value, np.mean(look_total_scores_window), np.std(look_total_scores_window)])

				#wave
				plot_value = wave/len_ep
				wave_total_scores_window.append(plot_value) 
				wave_total_scores.append([plot_value, np.mean(wave_total_scores_window), np.std(wave_total_scores_window)])

				#wave negative
				plot_value = wvneg/len_ep
				wave_negative_scores_window.append(plot_value) 
				wave_negative_scores.append([plot_value, np.mean(wave_negative_scores_window), np.std(wave_negative_scores_window)])

				#wave positive
				plot_value = wvpos/len_ep
				wave_positive_scores_window.append(plot_value) 
				wave_positive_scores.append([plot_value, np.mean(wave_positive_scores_window), np.std(wave_positive_scores_window)])

				#handshake
				plot_value = handshake/len_ep
				handshake_total_scores_window.append(plot_value) 
				handshake_total_scores.append([plot_value, np.mean(handshake_total_scores_window), np.std(handshake_total_scores_window)])


				#handshake positive
				plot_value = hspos/len_ep
				handshake_positive_scores_window.append(plot_value) 
				handshake_positive_scores.append([plot_value, np.mean(handshake_positive_scores_window), np.std(handshake_positive_scores_window)])

				#handshake negative
				plot_value = hsneg/len_ep
				handshake_negative_scores_window.append(plot_value) 
				handshake_negative_scores.append([plot_value, np.mean(handshake_negative_scores_window), np.std(handshake_negative_scores_window)])

				#handshake negative ratio
				if(hsneg+hspos==0):
					neg_plot_value = 0
					pos_plot_value = 0
				else:
					neg_plot_value = hsneg/(hsneg+hspos)
					pos_plot_value = hspos/(hsneg+hspos)

				handshake_ratio_negative_scores_window.append(neg_plot_value) 
				handshake_ratio_negative_scores.append([neg_plot_value, np.mean(handshake_ratio_negative_scores_window), np.std(handshake_ratio_negative_scores_window)])

				#handshake positive ratio
				handshake_ratio_positive_scores_window.append(pos_plot_value) 
				handshake_ratio_positive_scores.append([pos_plot_value, np.mean(handshake_ratio_positive_scores_window), np.std(handshake_ratio_positive_scores_window)])

				if(wvneg+wvpos==0):
					neg_plot_value = 0
					pos_plot_value = 0
				else:
					neg_plot_value = wvneg/(wvneg+wvpos)
					pos_plot_value = wvpos/(wvneg+wvpos)

				wave_ratio_negative_scores_window.append(neg_plot_value) 
				wave_ratio_negative_scores.append([neg_plot_value, np.mean(wave_ratio_negative_scores_window), np.std(wave_ratio_negative_scores_window)])

				#handshake positive ratio
				wave_ratio_positive_scores_window.append(pos_plot_value) 
				wave_ratio_positive_scores.append([pos_plot_value, np.mean(wave_ratio_positive_scores_window), np.std(wave_ratio_positive_scores_window)])
			

			#plot_value = len(rewards[i])
			#scores_window.append(plot_value)
			#scores.append([plot_value, np.mean(scores_window), np.std(scores_window)])

		labels2 = ['Neutral Emotion','Positive Emotion','Negative Emotion']
		l = labels2[int(l)-1]


		print('\rCalculating...', end="")
		#average_scores.append([calc_average_scores(scores,maxlen=100),l])
		wait_average_scores.append([calc_average_scores(wait_total_scores,maxlen=100),l])
		look_average_scores.append([calc_average_scores(look_total_scores,maxlen=100),l])
		wave_average_scores.append([calc_average_scores(wave_total_scores,maxlen=100),l])
		wave_positive_average_scores.append([calc_average_scores(wave_positive_scores,maxlen=100),l])
		wave_negative_average_scores.append([calc_average_scores(wave_negative_scores,maxlen=100),l])
		handshake_average_scores.append([calc_average_scores(handshake_total_scores,maxlen=100),l])
		handshake_positive_average_scores.append([calc_average_scores(handshake_positive_scores,maxlen=100),l])
		handshake_negative_average_scores.append([calc_average_scores(handshake_negative_scores,maxlen=100),l])
		handshake_ratio_positive_average_scores.append([calc_average_scores(handshake_ratio_positive_scores,maxlen=100),l])
		handshake_ratio_negative_average_scores.append([calc_average_scores(handshake_ratio_negative_scores,maxlen=100),l])
		wave_ratio_positive_average_scores.append([calc_average_scores(wave_ratio_positive_scores,maxlen=100),l])
		wave_ratio_negative_average_scores.append([calc_average_scores(wave_ratio_negative_scores,maxlen=100),l])

	print('\rLoaded!\t', end="")
	print('\n')

def main(save=False):
	folders = []
	labels = []
	#labels.append('Gray and Face States (OpenCV)')
	#Alternative
	#labels.append('Train after each Epoch')
	#folders.append('results/20211030_064156')

	#labels.append('Train after 4 Steps')
	#folders.append('results/20211103_084843')


	#labels.append('Gray and Face States')




	#labels.append("'Visible Face' States")
	#folders.append('results/20211115_185645')

	
	#labels.append('Only Gray State')
	#folders.append('results/20211107_055205')

	
	
	

	#labels.append('Only Face State')
	#folders.append('results/20211120_223307')
	
	#labels.append('Depth and Face States')
	#folders.append('results/20211126_054848')

	#20211202_230251
	#labels.append('Only Depth State')
	#folders.append('results/20211202_230251')

	'''
	labels.append("Fail Reward: Negative - Emotions")
	#75k rb
	folders.append('results/20220103_083937')
	'''
	#50k rb 20211229_034231
	#labels.append("Emotions 50k")
	#folders.append('results/20211229_034231')

	#labels.append("Fail Reward: Neutral - Emotions")
	#75k rb

	#folders.append('results/20220108_182042')
	



	#20220105_223402


	#labels.append("Only face state (no emotions). Sim with emotions")
	#folders.append('results/20220105_223402')


	#labels.append("Only face state (no emotions). Sim with emotions")
	#folders.append('results/20220115_101708')


	#labels.append("Only face state (no emotions). Sim with emotions 2")
	#folders.append('results/20220122_150759')

	
	
	#labels.append("Simulator X1: 64 batch size")
	#labels.append("SocialDQN: Culmulative Rewards")
	#folders.append('results/20220209_201938')
	#labels.append("Simulator X2: 128 batch size")
	#folders.append('results/20220208_040251')

	'''
	folders.append('results/20220324_172505')
	folders.append('results/20220327_121704')
	labels.append('SocialMDQN')
	labels.append('MDQN')
	'''

	#folders.append('results/20220330_155521')
	
	#folders.append('results/20220407_090416')
	#labels.append('MDQN x0.5')
	#folders.append('results/20220401_200543')
	#labels.append('MDQN x1.0')
	
	#folders.append('results/20220403_173206')
	

	#labels.append('MDQN')
	#folders.append('results/20220509_213720')
	#labels.append('MDQN with Emotions')
	#folders.append('results/20220512_110224')
	labels.append('SocialDQN')
	folders.append('results/20220514_030339')
	labels.append('SocialDQN 128 batch_size')
	folders.append('results/20220612_072614')
	

	calc_all_scores(folders,labels)
	params = PARAMETERS['SimDRLSR']
	if(save):
		folder_index_name = 1
		directory = os.path.join('utils','validate_tool_results',str(folder_index_name))
		while(True):
			if not os.path.exists(directory):
				os.makedirs(directory)
				break
			folder_index_name += 1 
			directory = os.path.join('utils','validate_tool_results',str(folder_index_name))
			

		compare_cumulative_rewards(folders,labels,save=True,save_location=directory)		
		plot(wave_average_scores,params,100,title='Wave',metric='Ratio (%)',save=True,save_location=directory)		
		plot(wave_positive_average_scores,params,100,title='Wave with positive Reward',metric='Ratio (%)',save=True,save_location=directory)
		plot(wave_negative_average_scores,params,100,title='Wave with negative Reward',metric='Ratio (%)',save=True,save_location=directory)
		plot(handshake_average_scores,params,100,title='Handshake',metric='Ratio (%)',save=True,save_location=directory)
		plot(handshake_positive_average_scores,params,100,title='Handshake with positive Reward',metric='Ratio (%)',save=True,save_location=directory)
		plot(handshake_negative_average_scores,params,100,title='Handshake with negative Reward',metric='Ratio (%)',save=True,save_location=directory)
		plot(handshake_ratio_positive_average_scores,params,100,title='Ratio Positive Handshake',metric='Ratio (%)',save=True,save_location=directory)
		plot(handshake_ratio_negative_average_scores,params,100,title='Ratio Negative Handshake',metric='Ratio (%)',save=True,save_location=directory)
		plot(wave_ratio_positive_average_scores,params,100,title='Ratio Positive Wave',metric='Ratio (%)',save=True,save_location=directory)
		plot(wave_ratio_negative_average_scores,params,100,title='Ratio Negative Wave',metric='Ratio (%)',save=True,save_location=directory)
		plot(look_average_scores,params,100,title='Look',metric='Ratio (%)',save=True,save_location=directory)
		plot(wait_average_scores,params,100,title='Wait',metric='Ratio (%)',save=True,save_location=directory)

		textfile = open(os.path.join(directory,"info.txt"), "w")
		for lb,fd in zip(labels,folders):
			textfile.write(lb + "\n")
			textfile.write(fd + "\n")
		
		textfile.close()
		print("Plots saved at: "+str(directory))

	else:
		while True:
			print("=========================")		
			print("|1 :Culmulative Rewards\t|")
			print("|2 :Wait\t\t|")
			print("|3 :Look\t\t|")
			print("|4 :Wave\t\t|")
			print("|5 :Wave Positive\t|")
			print("|6 :Wave Negative\t|")
			print("|7 :Handshake\t\t|")
			print("|8 :Handshake Pos\t|")
			print("|9 :Handshake Neg\t|")
			print("|W :Ratio HS Pos\t|")
			print("|E :Ratio HS Neg\t|")
			print("|R :Ratio Wv Pos\t|")
			print("|T :Ratio Wv Neg\t|")
			print("|0 :Exit\t\t|")
			print("=========================")
			with keyboard.Events() as events:
				# Block for as much as possible
				event = events.get(1e6)
				if event.key == keyboard.KeyCode.from_char('1'):
					compare_cumulative_rewards(folders,labels)
				elif event.key == keyboard.KeyCode.from_char('4'):
					plot(wave_average_scores,params,100,title='Wave',metric='Ratio (%)')
				elif event.key == keyboard.KeyCode.from_char('5'):
					plot(wave_positive_average_scores,params,100,title='Wave with positive Reward',metric='Ratio (%)')
				elif event.key == keyboard.KeyCode.from_char('6'):
					plot(wave_negative_average_scores,params,100,title='Wave with negative Reward',metric='Ratio (%)')
				elif event.key == keyboard.KeyCode.from_char('7'):
					plot(handshake_average_scores,params,100,title='Handshake',metric='Ratio (%)')
				elif event.key == keyboard.KeyCode.from_char('8'):
					plot(handshake_positive_average_scores,params,100,title='Handshake with positive Reward',metric='Ratio (%)')
				elif event.key == keyboard.KeyCode.from_char('9'):
					plot(handshake_negative_average_scores,params,100,title='Handshake with negative Reward',metric='Ratio (%)')
				elif event.key == keyboard.KeyCode.from_char('w'):
					plot(handshake_ratio_positive_average_scores,params,100,title='Ratio Positive Handshake',metric='Ratio (%)')
				elif event.key == keyboard.KeyCode.from_char('e'):
					plot(handshake_ratio_negative_average_scores,params,100,title='Ratio Negative Handshake',metric='Ratio (%)')
				elif event.key == keyboard.KeyCode.from_char('r'):
					plot(wave_ratio_positive_average_scores,params,100,title='Ratio Positive Wave',metric='Ratio (%)')
				elif event.key == keyboard.KeyCode.from_char('t'):
					plot(wave_ratio_negative_average_scores,params,100,title='Ratio Negative Wave',metric='Ratio (%)')

				elif event.key == keyboard.KeyCode.from_char('3'):
					plot(look_average_scores,params,100,title='Look',metric='Ratio (%)')
				elif event.key == keyboard.KeyCode.from_char('2'):
					plot(wait_average_scores,params,100,title='Wait',metric='Ratio (%)')
				elif event.key == keyboard.KeyCode.from_char('0'):
					break;			
				else:
					print('\nIncorrect key...')

def convert_one_hot_to_number(value):

	i = list(value).index(1)
	print(value)
	return i

def main2(save=True):
	folders = []
	labels = []
	
	#20220208_040251
	labels.append("1")
	#75k rb
	#folders.append('results/20220108_182042')
	#folders.append('results/20220214_052339')
	#folders.append('results/20220330_155521')
	#folders.append('results/20220330_155521')
	#folders.append('results/20220514_030339')
	#folders.append('results/20220514_030339')
	#folders.append('results/20220514_030339')
	#20220612_072614
	folders.append('results/20220717_045509')
	folders.append('results/20220717_045509')
	folders.append('results/20220717_045509')

	#mdqn = 20220509_213720
	#mdqn with emotions = 20220512_110224
	#SocialDQN = 20220514_030339


	labels.append("2")
	#75k rb
	#folders.append('results/20220108_182042')


	labels.append("3")
	#75k rb
	#folders.append('results/20220108_182042')

	#labels.append("Only face state (no emotions). Sim with emotions")
	#folders.append('results/20220115_101708')
	

	calc_with_emotions(folders,labels)
	params = PARAMETERS['SimDRLSR']
	if(save):
		folder_index_name = 1
		directory = os.path.join('utils','validate_tool_results',str(folder_index_name))
		while(True):
			if not os.path.exists(directory):
				os.makedirs(directory)
				break
			folder_index_name += 1 
			directory = os.path.join('utils','validate_tool_results',str(folder_index_name))
		
		scores = extract_cumulative_rewards_by_emotion(folders[0],'')
		params = PARAMETERS['SimDRLSR']
		#plot_culmulative(scores,'result',params,100,save=True,save_location=directory)		

		compare_cumulative_rewards(folders,labels,save=True,save_location=directory)	
		plot_culmulative(scores,'result',params,100,save=True,save_location=directory)	
		labels = ['Neutral Emotion','Positive Emotion','Negative Emotion']
		plot(wave_average_scores,params,100,title='Wave',metric='Ratio (%)',save=True,save_location=directory)		
		plot(wave_positive_average_scores,params,100,title='Wave with positive Reward',metric='Ratio (%)',save=True,save_location=directory)
		plot(wave_negative_average_scores,params,100,title='Wave with negative Reward',metric='Ratio (%)',save=True,save_location=directory)
		plot(handshake_average_scores,params,100,title='Handshake',metric='Ratio (%)',save=True,save_location=directory)
		plot(handshake_positive_average_scores,params,100,title='Handshake with positive Reward',metric='Ratio (%)',save=True,save_location=directory)
		plot(handshake_negative_average_scores,params,100,title='Handshake with negative Reward',metric='Ratio (%)',save=True,save_location=directory)
		plot(handshake_ratio_positive_average_scores,params,100,title='Ratio Positive Handshake',metric='Ratio (%)',save=True,save_location=directory)
		plot(handshake_ratio_negative_average_scores,params,100,title='Ratio Negative Handshake',metric='Ratio (%)',save=True,save_location=directory)
		plot(wave_ratio_positive_average_scores,params,100,title='Ratio Positive Wave',metric='Ratio (%)',save=True,save_location=directory)
		plot(wave_ratio_negative_average_scores,params,100,title='Ratio Negative Wave',metric='Ratio (%)',save=True,save_location=directory)
		plot(look_average_scores,params,100,title='Look',metric='Ratio (%)',save=True,save_location=directory)
		plot(wait_average_scores,params,100,title='Wait',metric='Ratio (%)',save=True,save_location=directory)

		textfile = open(os.path.join(directory,"info.txt"), "w")
		for lb,fd in zip(labels,folders):
			textfile.write(lb + "\n")
			textfile.write(fd + "\n")
		
		textfile.close()
		print("Plots saved at: "+str(directory))

	else:
		while True:
			print("=========================")		
			print("|1 :Culmulative Rewards\t|")
			print("|2 :Wait\t\t|")
			print("|3 :Look\t\t|")
			print("|4 :Wave\t\t|")
			print("|5 :Wave Positive\t|")
			print("|6 :Wave Negative\t|")
			print("|7 :Handshake\t\t|")
			print("|8 :Handshake Pos\t|")
			print("|9 :Handshake Neg\t|")
			print("|W :Ratio HS Pos\t|")
			print("|E :Ratio HS Neg\t|")
			print("|R :Ratio Wv Pos\t|")
			print("|T :Ratio Wv Neg\t|")
			print("|0 :Exit\t\t|")
			print("=========================")
			with keyboard.Events() as events:
				# Block for as much as possible
				event = events.get(1e6)
				if event.key == keyboard.KeyCode.from_char('1'):
					compare_cumulative_rewards(folders,labels)
				elif event.key == keyboard.KeyCode.from_char('4'):
					plot(wave_average_scores,params,100,title='Wave',metric='Ratio (%)')
				elif event.key == keyboard.KeyCode.from_char('5'):
					plot(wave_positive_average_scores,params,100,title='Wave with positive Reward',metric='Ratio (%)')
				elif event.key == keyboard.KeyCode.from_char('6'):
					plot(wave_negative_average_scores,params,100,title='Wave with negative Reward',metric='Ratio (%)')
				elif event.key == keyboard.KeyCode.from_char('7'):
					plot(handshake_average_scores,params,100,title='Handshake',metric='Ratio (%)')
				elif event.key == keyboard.KeyCode.from_char('8'):
					plot(handshake_positive_average_scores,params,100,title='Handshake with positive Reward',metric='Ratio (%)')
				elif event.key == keyboard.KeyCode.from_char('9'):
					plot(handshake_negative_average_scores,params,100,title='Handshake with negative Reward',metric='Ratio (%)')
				elif event.key == keyboard.KeyCode.from_char('w'):
					plot(handshake_ratio_positive_average_scores,params,100,title='Ratio Positive Handshake',metric='Ratio (%)')
				elif event.key == keyboard.KeyCode.from_char('e'):
					plot(handshake_ratio_negative_average_scores,params,100,title='Ratio Negative Handshake',metric='Ratio (%)')
				elif event.key == keyboard.KeyCode.from_char('r'):
					plot(wave_ratio_positive_average_scores,params,100,title='Ratio Positive Wave',metric='Ratio (%)')
				elif event.key == keyboard.KeyCode.from_char('t'):
					plot(wave_ratio_negative_average_scores,params,100,title='Ratio Negative Wave',metric='Ratio (%)')

				elif event.key == keyboard.KeyCode.from_char('3'):
					plot(look_average_scores,params,100,title='Look',metric='Ratio (%)')
				elif event.key == keyboard.KeyCode.from_char('2'):
					plot(wait_average_scores,params,100,title='Wait',metric='Ratio (%)')
				elif event.key == keyboard.KeyCode.from_char('0'):
					break;			
				else:
					print('\nIncorrect key...')

def find_files_candidates(filename,folder_content):

	candidates = [path for path in folder_content if (path.startswith(filename)and path.endswith('.csv'))]
	return candidates

def compare_cumulative_rewards(folders,labels,save=False,save_location=''):
	scores = []
	for fol,l in zip(folders,labels): 
		files = os.path.join(fol,'scores')
		folder_content = os.listdir(files)
		#filename = 'NeuralQLearner_simDRLSR'
		filename = 'MultimodalNeuralQLearner_simDRLSR'		
		candidates = find_files_candidates(filename,folder_content)
		if(len(candidates)==0):
			filename = 'NeuralQLearner_simDRLSR'		
			candidates = find_files_candidates(filename,folder_content)
		print(candidates)
		score = pd.read_csv(os.path.join(fol,'scores',candidates[0]), sep=',',nrows=MAX_EPS)
		scores.append([calc_average_scores(score['scores'],maxlen=250),l])

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
	plot_culmulative(scores,'result',params,100,save=save,save_location=save_location)


def extract_cumulative_rewards_by_emotion(folder,label):
	emotion_pth = os.path.join(folder,'scores','social_signals_history.dat')
	emotions = torch.load(emotion_pth)
	print(emotions)
	files = os.path.join(folder,'scores')
	folder_content = os.listdir(files)
	#filename = 'NeuralQLearner_simDRLSR_batch'
	filename = 'MultimodalNeuralQLearner_simDRLSR'		
	candidates = find_files_candidates(filename,folder_content)
	if(len(candidates)==0):
		filename = 'NeuralQLearner_simDRLSR'		
		candidates = find_files_candidates(filename,folder_content)
	print(candidates)

	#candidates = [path for path in folder_content if (path.startswith(filename)and path.endswith('.csv'))]
	score = pd.read_csv(os.path.join(folder,'scores',candidates[-1]), sep=',',nrows=MAX_EPS)
	print(candidates[-1])
	#scores.append([calc_average_scores(score['scores'],maxlen=250),label])
	n_emotions = 3
	dfs = []
	for n in range(n_emotions):
		dfs.append(pd.DataFrame(columns = score.columns))
	
	
	for i in range(len(emotions)):
		emotion_ep = 1
		for step in range(len(emotions[i])):		
			if(len(emotions[i])>0 ):
				emotion = emotions[i][step].numpy()
				n_emotion = convert_one_hot_to_number(emotion[0])
				if(n_emotion>0):
					emotion_ep = n_emotion
					break;
		row = score.iloc[i]
		dfs[emotion_ep-1]=dfs[emotion_ep-1].append(row,ignore_index = True)
	scores = []	
	labels = ['Neutral Emotion','Positive Emotion','Negative Emotion']
	for i in range(n_emotions):
		scores.append([calc_average_scores(dfs[i]['scores'],maxlen=250),labels[i]])
		print(len(scores[-1][0]))

	return scores	

def cumulative():
	folders = []
	labels = []
	#labels.append('Gray and Face States (OpenCV)')
	#Alternative
	#labels.append('Train after each Epoch')
	#folders.append('results/20211030_064156')

	#labels.append('Train after 4 Steps')
	#folders.append('results/20211103_084843')


	#labels.append('Gray and Face States')




	#labels.append("'Visible Face' States")
	#folders.append('results/20211115_185645')

	
	#labels.append('Only Gray State')
	#folders.append('results/20211107_055205')

	
	
	

	#labels.append('Only Face State')
	#folders.append('results/20211120_223307')
	
	#labels.append('Depth and Face States')
	#folders.append('results/20211126_054848')

	#20211202_230251
	#labels.append('Only Depth State')
	#folders.append('results/20211202_230251')

	'''
	labels.append("Fail Reward: Negative - Emotions")
	#75k rb
	folders.append('results/20220103_083937')
	'''
	#50k rb 20211229_034231
	#labels.append("Emotions 50k")
	#folders.append('results/20211229_034231')

	#labels.append("Fail Reward: Neutral - Emotions")
	#75k rb

	#folders.append('results/20220108_182042')
	



	#20220105_223402


	#labels.append("Only face state (no emotions). Sim with emotions")
	#folders.append('results/20220105_223402')


	#labels.append("Only face state (no emotions). Sim with emotions")
	#folders.append('results/20220115_101708')


	#labels.append("Only face state (no emotions). Sim with emotions 2")
	#folders.append('results/20220122_150759')

	
	
	#labels.append("Simulator X1: 64 batch size")
	labels.append("SocialDQN: Culmulative Rewards")
	#folders.append('results/20220209_201938')
	folders.append('results/20220214_052339')
	#labels.append("Simulator X2: 128 batch size")
	#folders.append('results/20220208_040251')
	
	

	
	params = PARAMETERS['SimDRLSR']

	folder_index_name = 1
	directory = os.path.join('utils','validate_tool_results',str(folder_index_name))
	while(True):
		if not os.path.exists(directory):
			os.makedirs(directory)
			break
		folder_index_name += 1 
		directory = os.path.join('utils','validate_tool_results',str(folder_index_name))
		

	compare_cumulative_rewards(folders,labels,save=True,save_location=directory)	

def main3(save=True):
	folders = []
	labels = []
	
	#20220208_040251
	labels.append("1")
	#75k rb
	#folders.append('results/20220108_182042')
	#folders.append('results/20220209_201938')
	folders.append('results/20220717_045509')
	folders.append('results/20220717_045509')
	folders.append('results/20220717_045509')

	labels.append("2")
	#75k rb
	#folders.append('results/20220108_182042')


	labels.append("3")
	#75k rb
	#folders.append('results/20220108_182042')

	#labels.append("Only face state (no emotions). Sim with emotions")
	#folders.append('results/20220115_101708')
	params = PARAMETERS['SimDRLSR']

	plot_culmulative(scores,'result',params,100,save=save,save_location=save_location)

if __name__ == "__main__":
	main(save=True)
	#main3()
	#cumulative()

