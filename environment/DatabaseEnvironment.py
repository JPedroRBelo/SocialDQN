import sys
sys.path.append('../')
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
import socket
import time
import io
from io import BytesIO
from PIL import Image
import signal
import subprocess
from subprocess import Popen
import os
import signal
from os.path import abspath, dirname, join
from utils.SocialSigns import SocialSigns
import pandas as pd
import select
from inspect import currentframe, getframeinfo
import torchvision.transforms.functional as TF
import cv2 
import cv2 as cv



class Environment:
	def __init__(self,params,path=''):
		# if gpu is to be used
		self.device = params['device']
		#self.r_len=8
		self.raw_frame_height= params['frame_height']
		self.raw_frame_width= params['frame_width']
		self.proc_frame_size= params['frame_size']
		self.state_size=params['state_size']
		self.neutral_reward = params['neutral_reward']
		self.hs_success_reward = params['hs_success_reward']
		self.hs_fail_reward = params['hs_fail_reward']
		self.eg_success_reward = params['eg_success_reward']
		self.eg_fail_reward = params['eg_fail_reward']
		self.ep_fail_reward = params['ep_fail_reward']
		self.ep_fail_reward = params['ep_fail_reward']
		self.use_depth_state = params['use_depth_state']
		self.blind_mode = params['blind_mode']
		self.social_state_size = params['social_state_size']
		self.use_only_depth_state = params['use_only_depth_state']
		self.emotional_states = params['emotional_states']
		self.facial_states = params['facial_states']
		series = pd.Series(self.emotional_states)
		if(self.social_state_size==2):
			series = pd.Series(self.facial_states)
		self.one_hot_vectors = pd.get_dummies(series)
		self.path = path
		self.params = params
		self.step = 0

		self.SocialSigns = SocialSigns()


		self.process = Popen('false')		


		self.ep_debug = 0




	def get_tensor_from_file(self,file):
		convert = T.Compose([T.ToPILImage(),
			T.Resize((self.proc_frame_size,self.proc_frame_size), interpolation=T.InterpolationMode.BILINEAR),
			T.ToTensor()])
		screen = Image.open(file)
		screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
		screen = torch.from_numpy(screen)
		screen = convert(screen).to(self.device)
		return screen

	'''
	def pre_process(self,step):	
		proc_image=torch.FloatTensor(self.state_size,self.proc_frame_size,self.proc_frame_size)
		proc_depth=torch.FloatTensor(self.state_size,self.proc_frame_size,self.proc_frame_size)
		
		dirname_rgb='dataset/RGB/ep'+str(self.episode)
		dirname_dep='dataset/Depth/ep'+str(self.episode)
		for i in range(self.state_size):

			grayfile=dirname_rgb+'/image_'+str(step)+'_'+str(i+1)+'.png'
			depthfile=dirname_dep+'/depth_'+str(step)+'_'+str(i+1)+'.png'
			proc_image[i] = self.get_tensor_from_file(grayfile)
			proc_depth[i] = self.get_tensor_from_file(depthfile)			

		return proc_image.unsqueeze(0),proc_depth.unsqueeze(0)
	'''

	def get_one_hot_vector(self,label):
		return self.one_hot_vectors[label].values

	def get_tensor_from_image(self,screen):
		convert = T.Compose([T.ToPILImage(),
			T.Resize((self.proc_frame_size,self.proc_frame_size), interpolation=T.InterpolationMode.BILINEAR),
			T.ToTensor()])
		screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
		screen = torch.from_numpy(screen)
		screen = convert(screen).to(self.device)
		return screen



	def convert_to_image(self,bytes_image):

		dataBytesIO = BytesIO(bytes_image)
		try:
			img = Image.open(dataBytesIO).convert('L')
			
		except:
			#img =  Image.new('L', (self.proc_frame_size, self.proc_frame_size))
			#print('Image Error...')
			return None

		return img

	def pre_process(self,images):
		proc_image=torch.FloatTensor(self.state_size,self.proc_frame_size,self.proc_frame_size)		
		i = 0
		for image in images:
			proc_image[i] = self.get_tensor_from_image(image)	
			i += 1		
		return proc_image.unsqueeze(0)

	def is_final_state(self,action,reward):
		if(action=='4') and (reward == self.hs_success_reward):

			return True
		else:
			return False


	def get_screen(self,ep):
		
		states_gray = []
		states_depth = []
		s = []
		d = []
		face_count = []	

		for i in range(self.state_size):
			size = 0
			path_image = os.path.join(self.path,'images','2')+"/gray"+str(ep)+"_"+str(i)+".png"
			image = Image.open(path_image)
			#image_array = np.array(image)
			#image = np.moveaxis(image_array, -1, 0)
			im_l = image.convert('L')
			#image = self.convert_to_image(image)
			states_gray.append(im_l)
				

		emotion = np.load(os.path.join(self.path,'scores')+'/social_signals_history.npy')[ep]
		emotion = self.emotional_states[emotion]
		emotion_one_hot = self.get_one_hot_vector(emotion)
		face_state = torch.FloatTensor(emotion_one_hot).unsqueeze(0)

		s = self.pre_process(states_gray)
		d = None
		s = [s,face_state]
		return s,d

	def most_common(self,lst):
		return max(set(lst), key=lst.count)

	def receive_image(self,size):
		read = 0
		while True:
			self.socket.settimeout(self.socket_time_out)	
			recv = self.socket.recv(size)
			read += len(recv)
			if(read>=size):				
				break
		return recv	

	
	def close_connection(self):
		self.close_simulator()	
		self.socket.close()

	def close(self):
		self.socket.close()

	def init_simulator(self,command):
		self.process = self.openSim(command,self.process)

	def close_simulator(self):
		self.killSim(self.process)



	def openSim(self,command,process):
		process.terminate()
		process = Popen(command, shell=True, preexec_fn=os.setsid)
		return process

	def killSim(self,process):
		process.terminate()
		os.killpg(os.getpgid(process.pid), signal.SIGTERM)		
		#time.sleep(1)

	def signalHandler(self,sig, frame):
	    self.process.terminate()
	    sys.exit(0)

