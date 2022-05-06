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



class Environment:
	def __init__(self,params,simulator_path='',start_simulator=False,verbose=False,epi=0,port=0):
		# if gpu is to be used
		self.device = params['device']
		#self.r_len=8
		self.episode=epi
		self.verbose = verbose
		self.raw_frame_height= params['frame_height']
		self.raw_frame_width= params['frame_width']
		self.proc_frame_size= params['frame_size']
		self.state_size=params['state_size']
		self.simulation_speed = params['simulation_speed']
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
		self.emotion_type= params['emotion_type']
		self.robot_random_position = params['robot_random_position']
		self.human_appearance = params['human_appearance']
		series = pd.Series(self.emotional_states)
		if(self.social_state_size==2):
			series = pd.Series(self.facial_states)
		self.one_hot_vectors = pd.get_dummies(series)
		
		self.params = params
		self.step = 0
		self.socket_time_out = params['socket_time_out']
		self.SocialSigns = SocialSigns()
		self.port = self.params['port']
		if(port!=0):
			self.port = port
		self.process = Popen('false')		
		signal.signal(signal.SIGINT, self.signalHandler)
		self.debug= False
		self.ep_debug = 0

		if(start_simulator):
			name = params['env_name']
			folder = name
			if(not simulator_path==''):
				folder = simulator_path			
			
			command = './'+name+'.x86_64'			
			command = abspath(join(folder,command))
			par = ' -screen-width '+str(params['screen_width'])
			par += ' -screen-height '+str(params['screen_height'])
			par += ' -port '+str(port)
			self.init_simulator(command+par)
		self.socket,self.client = self.__connect()	
		self.set_configuration()

	def setDebug(self,ep_debug):
		self.debug = True
		self.ep_debug = ep_debug

	def __connect(self):
		skt = socket.socket(socket.AF_INET, socket.SOCK_STREAM)	
		host=self.params['host']
		flag_connection = False	
		count = 0	
		while(not flag_connection):
			try:
				client =skt.connect((host, self.port))
				flag_connection = True
				return skt,client
			except socket.error:
				if count > 5:
					print("Can't connect with robot! Trying again...")
				count += 1
				time.sleep(1)

		return None,None
	def set_configuration(self):
		self.config_simulation("speed"+str(self.simulation_speed))
		self.config_simulation("reward neutral:"+str(self.neutral_reward))
		self.config_simulation("reward hs_success:"+str(self.hs_success_reward))
		self.config_simulation("reward hs_fail:"+str(self.hs_fail_reward))
		self.config_simulation("reward eg_success:"+str(self.eg_success_reward))
		self.config_simulation("reward eg_fail:"+str(self.eg_fail_reward))
		self.config_simulation("use_depth"+str(self.use_depth_state))
		self.config_simulation("emotion_type"+str(self.emotion_type))
		self.config_simulation("robot_random_position"+str(self.robot_random_position))
		self.config_simulation("human_appearance"+str(self.human_appearance))

	def connect(self):
		self.socket,self.client = self.__connect()
		timeout = 5
		ready_sockets, _, _ = select.select([self.socket], [], [], timeout)
		if ready_sockets:
			data = self.socket.recv(1024)
			print("Junk: "+str(data))


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

	
	def send_data_to_pepper(self,data):
		action = self.params['actions'][data]
		self.socket.send(action.encode())
		while True:
			self.socket.settimeout(self.socket_time_out )
			reward = self.socket.recv(1024).decode()
			if reward:
				try:
					reward_value = float(reward.replace(',','.'))
					print(reward_value)
					return reward_value
				except (ValueError, TypeError):
					print("Reward error")
					continue				
			break
		return 0

	def config_simulation(self,data,text="Configuring"):

		if self.verbose: 
			print('{} Simulator: {}'.format(text,data))
		done = False;
		while not done:
			try: 
				self.socket.send(data.encode())
			except Exception:
				print("Connection Exception")
				return 0

			while (True):
				self.socket.settimeout(self.socket_time_out)
				msg = self.socket.recv(1024)
				msg = msg.decode()
				if msg:
					return float(msg.replace(',','.').replace('\n',''))

				break


		return 0

	def reseting_simulation(self):

		data = "reset"
		done = False;
		while not done:
			try: 
				self.socket.send(data.encode())
			except Exception:
				print("Connection Exception")
				return -1

			while True:
				
				self.socket.settimeout(self.socket_time_out)		
				msg = self.socket.recv(1024)			
				msg = msg.decode()
				done = True
				return 1


		return 0

	def is_final_state(self,action,reward):
		if(action=='4') and (reward == self.hs_success_reward):

			return True
		else:
			return False


	def execute(self,data):
		action = self.params['actions'][data]
		self.socket.send(action.encode())
		terminal = False
		
		while True:
			self.socket.settimeout(self.socket_time_out)
			data = self.socket.recv(1024).decode()
			if data:
				if("reward" in data):
					original_data = str(data)
					data = data. replace("reward", "")
					data = data. replace(" ", "")
					reward = float(data.replace(',','.'))
					terminal = self.is_final_state(action,reward)
					if self.step >= self.params['t_steps']-1:
						terminal = True
						reward = self.ep_fail_reward
					#print("Ep: "+str(self.episode)+" step: "+str(self.step))
					self.step += 1
					#print("Reward: "+str(original_data)+ "converted "+str(reward))
					return reward,terminal	
	
	
		return 0

	'''
	def perform_action(self,action,step):
		r=self.send_data_to_pepper(action)
		s,d=self.pre_process(step)
		term = False
		return s,d,r,term

	'''

	def get_screen(self):
		
		states_gray = []
		states_depth = []
		s = []
		d = []
		face_count = []
		self.socket.send('get_screen'.encode())
		while True:		
			self.socket.settimeout(self.socket_time_out)
			recv = self.socket.recv(1024)
			if(recv):
				break;
		counter = 0
		import struct
		j = 0
		n_channels = 1
		if(self.use_depth_state):
			n_channels+=1



		for i in range(n_channels*self.state_size):
			size = 0
			self.socket.send('next_size'.encode())
			while True:	
				self.socket.settimeout(self.socket_time_out)						
				recv = self.socket.recv(6)
				recv = recv.decode().rstrip("\n")
				if recv.isdigit():
					size = int(recv)#.decode()
					if size != 0:
						break;
			#self.socket.send('next'.encode())

			self.socket.send('next_image'.encode())		
			data_img = self.receive_image(size)
			if(not self.blind_mode):
				image = self.convert_to_image(data_img)
			else:
				image =  Image.new('L', (self.proc_frame_size, self.proc_frame_size))
			n_tries = 0
			while True:
				if(image == None):
					n_tries += 1
					#print("Image error: ",str(i))
					self.socket.send('last_image'.encode())
					data_img= self.receive_image(size)
					image = self.convert_to_image(data_img)
					if(n_tries>3):
						pass
						#print("Image {} Error: #{}th attempt.".format(i,n_tries))
				else:
					break


			if(counter%n_channels==0):
				#num_faces = self.SocialSigns.find_faces(image)
				#face= num_faces
				self.socket.send('next_emotion'.encode())
				face = 'no_face'
				while True:
					
					self.socket.settimeout(self.socket_time_out)
					msg = self.socket.recv(1024).decode()
					if msg:				
						face = msg.replace('\n','')
					break

				'''
				if(face in self.emotional_states):
					face_count.append(face)
				'''
				if(self.social_state_size==2):
					if(face in self.emotional_states):
						aux = min(self.emotional_states.index(face),(self.social_state_size-1))

						face_count.append(self.facial_states[aux])
						
					elif(face in self.facial_states):
						face_count.append(face)

				else:
					if(face in self.emotional_states):
						face_count.append(face)
				states_gray.append(image)

			else:
				
				states_depth.append(image)
				
			counter += 1
		emotion = self.most_common(face_count)
		emotion_one_hot = self.get_one_hot_vector(emotion)
		face_state = torch.FloatTensor(emotion_one_hot).unsqueeze(0)
		s = self.pre_process(states_gray)
		d = None
		if(self.use_depth_state):
			d = self.pre_process(states_depth)
		if(self.use_only_depth_state):
			s = d
		if(d != None):
			d = [d,face_state]
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

	def reset(self,restart_simulator=True):
		self.step = 0
		if(restart_simulator):
			result = self.reseting_simulation()
			if(result==-1):
				print("Can't connect with simulator")
				return 0
			elif(result == 0):
				self.reseting_simulation()
			else:
				time.sleep(3)
				self.close()
				self.connect()				
				self.set_configuration()
				return 1


	
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

