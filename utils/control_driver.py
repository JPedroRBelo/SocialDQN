# -*- coding: utf-8 -*-
import sys
sys.path.append('../')
import torch
import torch.nn as nn
from pathlib import Path
from environment.SimEnvironment import Environment
from pynput import keyboard

from config.controldriverparams import *
params = PARAMETERS['SimDRLSR']
env_name = params['env_name']




def generate_data(episode,env):
	env = Environment(params)
	t_steps = 2000
	total_reward = 0

	aset = ['1','2','3','4']

	env.close_connection()
	env = Environment(params)


	reward = 0 #temp
	terminal = 0
	screen = None
	depth = None
	init_step = 0

	step=init_step+1
	while step <=t_steps+1:
		print("Step=",step)
		action_index=0
		
		
		print("1 :Wait\n2 :Look\n3: Wave\n4: Handshake\n")
		with keyboard.Events() as events:
			# Block for as much as possible
			event = events.get(1e6)
			if event.key == keyboard.KeyCode.from_char('1'):
				action_index = 1	
			elif event.key == keyboard.KeyCode.from_char('2'):
				action_index = 2	
			elif event.key == keyboard.KeyCode.from_char('3'):
				action_index = 3	
			elif event.key == keyboard.KeyCode.from_char('4'):
				action_index = 4				
			else:
				print('\nIncorrect key... sending "Wait" command to robot!')
				action_index = 1

		
		action_index = int(action_index)-1
		step=step+1		
		reward, done = env.execute(action_index)

		#rewards.append(reward)
		#actions.append(action_index)
		total_reward=total_reward+reward
		print("Total Reward: ",total_reward)
		print('================>')
		#torch.save(rewards,'recent_rewards.dat',)
		#torch.save(actions,'recent_actions.dat')

		

	#torch.save([],'recent_rewards.dat')
	#torch.save([],'recent_actions.dat')

	

def main():
	#tracker = SummaryTracker()
	episode="ControlDriver"
	dirname_rgb='dataset/RGB/ep'+str(episode)
	dirname_dep='dataset/Depth/ep'+str(episode)
	dirname_model='results/ep'+str(episode)


	
	env = Environment(params)

	Path(dirname_rgb).mkdir(parents=True, exist_ok=True)
	Path(dirname_dep).mkdir(parents=True, exist_ok=True)
	Path(dirname_model).mkdir(parents=True, exist_ok=True)

	generate_data(episode,env)
	env.close_connection()

if __name__ == "__main__":
   main()
