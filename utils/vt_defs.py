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


def plot(scores,params,i_episode,save=False,title='',metric='Score'):


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
	ax.legend(legends,loc='lower right')
		
	#plt.fill_between(episode,df['average_scores'].add(df['std']),df['average_scores'].sub(df['std']),alpha=0.3)

	plt.title(title)
	plt.ylabel(metric)
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




def plot_rewards(scores,name,params,i_episode,save=False):


	fig = plt.figure(num=2,figsize=(10, 5))
	plt.clf()

	plt.legend(loc='lower right')
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



def plot_culmulative(scores,name,params,i_episode,save=False):


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
	ax.legend(legends,loc='lower right')
		
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
		pth = os.path.join(fol,'scores/action_reward_history.dat')
		rewards = torch.load(pth)
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


def compare_actions(folders,labels,arg):
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
		files = os.path.join(fol,'scores')
		folder_content = os.listdir(files)
		filename = 'NeuralQLearner_simDRLSR_batch_128_lr_3E-04_trained'
		candidates = [path for path in folder_content if (path.startswith(filename)and path.endswith('.csv'))]
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
	plot_culmulative(scores,'result',params,100)

def main():
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

	labels.append("Fail Reward: Negative - Emotions")
	#75k rb
	folders.append('results/20220103_083937')
	#50k rb 20211229_034231
	#labels.append("Emotions 50k")
	#folders.append('results/20211229_034231')

	labels.append("Fail Reward: Netral - Emotions")
	#75k rb
	folders.append('results/20220108_182042')


	
	

	calc_all_scores(folders,labels)
	params = PARAMETERS['SimDRLSR']

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


def main2():
	pass

if __name__ == "__main__":
	main2()