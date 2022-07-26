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
    df = pandas.DataFrame(scores,columns=['scores','average_scores','std'])
    fig = plt.figure(num=2,figsize=(10, 5))
    plt.clf()


    ax = fig.add_subplot(111)
    episode = np.arange(len(scores))
    plt.plot(episode,df['average_scores'])
    plt.fill_between(episode,df['average_scores'].add(df['std']),df['average_scores'].sub(df['std']),alpha=0.3)

    ax.legend([' [ Average scores ]'])
    plt.ylabel('Score')
    plt.xlabel('Episode')
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

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 


rewards=torch.load(parentdir+'/scores/action_reward_history.dat')#.detach().cpu().numpy()

""" Training loop  """
scores = []                                 # list containing scores from each episode
scores_window = deque(maxlen=100)   # last (window_size) scores

arg = 'all'
if len(sys.argv) > 1:
	arg = sys.argv[1]

v_hspos = []
v_hsneg = []
v_wvpos = []
v_wvneg = []
v_wave  = []
v_wait  = []
v_look  = []
ep_durations = []

params = PARAMETERS['SimDRLSR']
neutral_reward = params['neutral_reward']
hs_success_reward = params['hs_success_reward']
hs_fail_reward = params['hs_fail_reward']
eg_success_reward = params['eg_success_reward']
eg_fail_reward = params['eg_fail_reward']
ep_fail_reward = params['ep_fail_reward']

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
	
	ep_durations
	
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

plot()

'''
if (arg == 'hs') or (arg == 'all') or (arg == 'reward'):
	plt.plot(v_hspos,label='HS Success')
	plt.plot(v_hsneg,label='HS Fail')

if (arg == 'ep_duration'):
	plot()

if (arg == 'wv') or (arg == 'all')  or (arg == 'reward'):
	plt.plot(v_wvpos,label='Wave Success')
	plt.plot(v_wvneg,label='Wave Fail')
	
if (arg == 'other') or (arg == 'all'):
	plt.plot(np.add(v_hspos, v_hsneg),label='HS Total')
	plt.plot(v_wait,label='Wait')
	plt.plot(v_look,label='Look')
	plt.plot(v_wave,label='Wave')


plt.ylabel('Number of Actions')
plt.xlabel('Epoch')

plt.ylim([0, 35])
plt.legend()
plt.show()
'''
