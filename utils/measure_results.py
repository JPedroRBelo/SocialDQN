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
import importlib.util


MAX_EPS = 15000


'''
def main(save=False):
	folders = []
	labels = []
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
'''

def find_files_candidates(filename,folder_content):

	candidates = [path for path in folder_content if (path.startswith(filename)and path.endswith('.csv'))]
	return candidates

def calc_average_scores(old_scores,maxlen=100):
	scores = []								 # list containing scores from each episode
	scores_window = deque(maxlen=maxlen)   # last (window_size) scores
	for s in old_scores:
		scores_window.append(s)
		scores.append([s, np.mean(scores_window), np.std(scores_window)])
	return scores

def get_scores(folder,label):



	files = os.path.join(folder,'scores')
	folder_content = os.listdir(files)
	#filename = 'NeuralQLearner_simDRLSR_batch_128_lr_3E-04_trained'
	#filename = 'NeuralQLearner_simDRLSR_batch'
	filename = 'MultimodalNeuralQLearner_simDRLSR'		
	candidates = find_files_candidates(filename,folder_content)
	if(len(candidates)==0):
		filename = 'NeuralQLearner_simDRLSR'		
		candidates = find_files_candidates(filename,folder_content)
	#candidates = [path for path in folder_content if (path.startswith(filename)and path.endswith('.csv'))]
	score = pd.read_csv(os.path.join(folder,'scores',candidates[0]), sep=',',nrows=MAX_EPS)
	scores  = [calc_average_scores(score['scores'],maxlen=100),label]
	return scores




def compare_dicts(d1, d2,l1,l2):
	print(f'\t {l1}\t {l2} ')
	for key in d1:

		if(d1[key]!=d2[key]):
			print(f'Key:  {key} \t {d1[key]} \t {d2[key]}')

def main(save=False):
	folders = []
	labels = []
	labels.append('20220709_192051')
	folders.append('results/20220709_192051')

	labels.append('20220708_124340')
	folders.append('results/20220708_124340')

	parameters = []
	scores = []
	scores_df = []
	maxlen = math.inf
	for lb,fl in zip(labels,folders):

		spec=importlib.util.spec_from_file_location("cfg",os.path.join(fl,"hyperparams.py"))
		cfg =  importlib.util.module_from_spec(spec)
		spec.loader.exec_module(cfg)
		params = cfg.PARAMETERS['SimDRLSR']
		#print(params)
		parameters.append(params)
		score = get_scores(fl,lb)
		scores.append(score)
		df = pandas.DataFrame(score,columns=['scores','average_scores','std'])
		scores_df.append(df)
		maxlen = min(maxlen,len(df['scores']))

	#compare_dicts(parameters[0],parameters[1],labels[0],labels[1])


if __name__ == "__main__":
	main(save=True)