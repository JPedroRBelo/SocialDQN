import sys
sys.path.append('../')
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
import time
from time import sleep, perf_counter
from io import BytesIO
from PIL import Image
import os
from os.path import abspath, dirname, join
from inspect import currentframe, getframeinfo
import pepperparams as cfg  
from deepface import DeepFace
import zmq
import json
import base64
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing.connection import Client, Connection, _ForkingPickler
from social.face_info import FaceDetection


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_blue(text):
	print(bcolors.OKCYAN + str(text) + bcolors.ENDC)

def send_py2(self, obj):
    self._check_closed()
    self._check_writable()
    self._send_bytes(_ForkingPickler.dumps(obj, protocol=2))

Connection.send = send_py2

class Environment:
	def __init__(self,params,verbose=False,epi=0,port=0):
		# if gpu is to be used
		self.device = params['device']
		self.robot_params = cfg.PARAMETERS['SimDRLSR']  
		#self.r_len=8
		self.episode=epi
		self.verbose = verbose
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
		self.emotion_type= params['emotion_type']
		series = pd.Series(self.emotional_states)
		if(self.social_state_size==2):
			series = pd.Series(self.facial_states)
		self.one_hot_vectors = pd.get_dummies(series)
		self.face = FaceDetection()
		
		self.params = params
		self.step = 0
		#kQQVGA (160x120), kQVGA (320x240),
		#kVGA (640x480) or k4VGA (1280x960, only with the HD camera).
		self.resolution = "kQVGA"
		#self.resolution = "kQQVGA"

		self.address_name = "pepper.local"
		#self.address_name = "localhost"
		self.bridge_port = 6666
		#socket or multiprocessing
		self.connection_method = "multiprocessing"


		if(self.connection_method=="multiprocessing"):
			address = (self.address_name, self.bridge_port)
			self.bridge_socket =  Client(address)
		else:
			self.context = zmq.Context()
			self.bridge_socket = self.context.socket(zmq.REP)
			self.bridge_socket.bind("tcp://*:"+str(self.bridge_port))

		#self.process = self.init_NAOqi_bridge(self.bridge_port)
		#connected = self.receive()
		#print("Connected: "+str(connected))
		#print(connected)
		#self.process = Popen('false')		
		
		self.debug= False
		self.ep_debug = 0
		self.set_configuration()

	def setDebug(self,ep_debug):
		self.debug = True
		self.ep_debug = ep_debug

		return None,None
	def set_configuration(self):
		self.config_robot("resolution:"+str(self.resolution))
		self.config_robot("reward neutral:"+str(self.neutral_reward))
		self.config_robot("reward hs_success:"+str(self.hs_success_reward))
		self.config_robot("reward hs_fail:"+str(self.hs_fail_reward))
		self.config_robot("reward eg_success:"+str(self.eg_success_reward))
		self.config_robot("reward eg_fail:"+str(self.eg_fail_reward))

	#Then specify the resolution among : kQQVGA (160x120), kQVGA (320x240),
	#kVGA (640x480) or k4VGA (1280x960, only with the HD camera).
	def get_resolution_values(self,resolution_name):
		if(resolution_name == 'kQQVGA'):
				return (160,120)
		elif(resolution_name == 'kQVGA'):
				return (320,240)
		elif(resolution_name == 'kVGA'):
				return (640,480)
		elif(resolution_name =='k4VGA'):
				return (1280,960)
		else:
			return None,None




	def get_tensor_from_file(self,file):
		convert = T.Compose([T.ToPILImage(),
			T.Resize((self.proc_frame_size,self.proc_frame_size), interpolation=T.InterpolationMode.BILINEAR),
			T.ToTensor()])
		screen = Image.open(file)
		screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
		screen = torch.from_numpy(screen)
		screen = convert(screen).to(self.device)
		return screen


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
	


	def send(self,data):
		if(self.connection_method=="multiprocessing"):
			self.bridge_socket.send(json.dumps(data))
		else:
			self.bridge_socket.send_json(json.dumps(data))

	def receive(self):
		if(self.connection_method=="multiprocessing"):
			message = self.bridge_socket.recv()
		else:
			message = self.bridge_socket.recv_json()
		message = json.loads(message)
		print("Received message")
		return message

	def config_robot(self,data,text="Configuring"):
		if self.verbose: 
			print('{} Robot: {}'.format(text,data))
		self.send(data)

		print("Receiving")
		message = self.receive()
		print("Received: "+str(message))
		return message


	def is_final_state(self,action,reward):
		if(action=='4') and (reward == self.hs_success_reward):

			return True
		else:
			return False


	def execute(self,data):
		action = self.params['actions'][data]

		self.send(action)
		terminal = False
		message = self.receive()
		print("After action: "+message)


		if("reward" in message):
			original_data = str(message)
			message = message.replace("reward", "")
			message = message.replace(" ", "")
			reward = float(message.replace(',','.'))
			terminal = self.is_final_state(action,reward)
			if self.step >= self.params['t_steps']-1:
				terminal = True
				reward = self.ep_fail_reward
			#print("Ep: "+str(self.episode)+" step: "+str(self.step))
			self.step += 1
			#print("Reward: "+str(original_data)+ "converted "+str(reward))
			return reward,terminal		
	
		return 0

	def to_byte_array(self,image):
		image = Image.frombytes("L", (320, 240), str(image))
		imgByteArr = io.BytesIO()
		image.save(imgByteArr, 'png')
		imgByteArr = imgByteArr.getvalue()
		return imgByteArr

	def convert_image(self,message):
		message = json.loads(message)			
		message = base64.b64decode(message)
		#message = BytesIO(message)
		#message = Image.open(message).convert('L')
		resolution = self.get_resolution_values(self.resolution)
		message = Image.frombytes("L", resolution, message)

		#message = json.loads(message)
		#message.show()
		return message

	def show_image(self,frame,rects,title="Image"):
		frameHeight = frame.shape[0]
		frameWidth = frame.shape[1]
		show = False
		for r in rects:
			show = True
			x1, y1, x2, y2 = r
			cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
		# Display
		if(show):
			cv2.imshow(title, frame)
			# Stop if escape key is pressed
			cv2.waitKey(0)

			# cv2.destroyAllWindows() simply destroys all the windows we created.
			cv2.destroyAllWindows()


	def get_images(self,index):
		states_emotion = []
		states_gray = []
		s = []
		emotion_count = []
		start_time = perf_counter()
		self.send('get_screen')
		#Wait previous message confirmation
		images = self.receive()
		#images = json.load(open(file_name))
		#images = json.loads(images)
		end_time = perf_counter()

		print(f'It took {end_time- start_time: 0.2f} second(s) to complete.')
		#print('received ')
		counter = 0
		count = 0
		detect_start_time = perf_counter()
		for i in images:
			image = self.convert_image(i)

			
			#save_path = "Image/image_"+str(index)+"_"+str(count)+".png"
			save_path = ''
			if(count%3==0): 
				#stretching , equalization , adaptative
				emotion = self.face.recognize_face_emotion_test(image=image,preprocess='adaptative',save_path=save_path)
				emotion_count.append(emotion)	

			states_gray.append(image)
			count += 1
				
		print("Step =>"+str(index))
		detect_end_time = perf_counter()

		emotion = self.face.choose_emotion_by_conf(emotion_count)
		group_emotion = self.emotion_to_group(emotion)
		print(f'{emotion} to {group_emotion}')
		emotion_one_hot = self.get_one_hot_vector(group_emotion)
		face_state = torch.FloatTensor(emotion_one_hot).unsqueeze(0)
		
		#group_emotion = emotion
		print(f'Emotion Detection Time: {detect_end_time- detect_start_time: 0.2f} second(s)')
		s = self.pre_process(states_gray)

		s = [s,face_state]
		return s




	def most_common(self,lst):
		return max(set(lst), key=lst.count)



	def close_connection(self):
		self.send('close_socket')
		self.bridge_socket.close()

	def close(self):
		self.close_connection()


if __name__ == "__main__":
		

	import hyperparams as cfg
	params = cfg.PARAMETERS['SimDRLSR']
	env = Environment(params,verbose=True)
	actions = params['actions_names']
	
	act = actions[:]
	#print(act)

	try:
		i_action = 0
		while(True):
			

			#action = actions.index('Handshake')
			#action = actions.index(a)
			#print("Executing action:"+str(i_action%4))
			#reward, done = env.execute(i_action%4)
			#print("Reward: "+str(reward))
			i_action += 1 
			print("Getting states")
			images = env.get_images(i_action)
			print_blue("Emotion: "+str(images[1]))
			#emotion_recogntion(images[0][0])
			#from torchvision.utils import save_image
			#for i in range(len(images[0])):
			#	save_image(images[0][i],"image_"+str(i)+".png")
			#if(i_action>=10):
			#	break
	except KeyboardInterrupt:	
		env.close_connection();



	
	'''
	action = 3
	reward, done = env.execute(action)
	action = 2
	reward, done = env.execute(action) 
	action = 3
	reward, done = env.execute(action)  
	'''
	#while(True):
	#	env.
	#	time.sleep(1)