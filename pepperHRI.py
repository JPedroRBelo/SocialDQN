import qi
import argparse
import functools
import sys, os
import socket
import numpy as np
import time
import cv2 as cv
import base64
from random import randrange
try:
	from PIL import Image
except ImportError:
	import Image
import vision_definitions as vd
import io
import json
import pickle
from multiprocessing.connection import Listener
import fcntl
import struct



class PepperHRI(object):
	""" A simple module able to react
		to touch events.
	"""
	def __init__(self, session,host):
		super(PepperHRI, self).__init__()
		self.session = session
		self.memory_service = session.service("ALMemory")
		self.motion_service = session.service("ALMotion")
		self.moodService = session.service("ALMood")
		self.tts_service = session.service("ALTextToSpeech")
		self.leds_service = session.service("ALLeds")

		# Connect to an Naoqi1 Event.
		self.touch = self.memory_service.subscriber("TouchChanged")
		self.id = self.touch.signal.connect(functools.partial(self.onTouched, "TouchChanged"))
		#self.memory_service.subscribe("PepperHRI")
		self.dict_sensors = {'RArm':False}
		#self.motion_service
		
		self.n_images = 8
		self.fps = 8
		self.neutral_reward = 0
		self.hs_success_reward = 1
		self.hs_fail_reward = -0.2
		self.eg_fail_reward = 0
		self.eg_success_reward = 0




		self.image_width = 320
		self.image_height = 240

		#Then specify the resolution among : kQQVGA (160x120), kQVGA (320x240),
		#kVGA (640x480) or k4VGA (1280x960, only with the HD camera).
		self.camProxy_service = session.service("ALVideoDevice")
		#resolution = 1	# VGA
		resolution = vd.kVGA
		self.colorSpace = 0   # Y channel

		self.upper_cam = self.camProxy_service.subscribeCamera("Ucam",0, resolution, self.colorSpace, 20)
		#self.depth = self.camProxy_service.subscribeCamera("Dcam",2, resolution, colorSpace, 5)

		self.basic_awareness_service = session.service("ALBasicAwareness")

		self.tracker_service = session.service("ALTracker")
		self.tablet_service = session.service("ALTabletService")

		self.set_image_lar()
		self.basic_awareness_service.setStimulusDetectionEnabled("People",True)
		self.basic_awareness_service.setStimulusDetectionEnabled("Movement",True)
		self.basic_awareness_service.setStimulusDetectionEnabled("Sound",True)
		self.basic_awareness_service.setStimulusDetectionEnabled("Touch",True)

		self.basic_awareness_service.setParameter("LookStimulusSpeed",0.7)
		self.basic_awareness_service.setParameter("LookBackSpeed",0.5)
		self.basic_awareness_service.setEngagementMode("FullyEngaged")
		self.basic_awareness_service.setTrackingMode("Head")

		self.images_to_send = []
		self.images_sended = True
		self.index_image_to_send = 0

		self.targetName = "Face"
		self.faceWidth = 0.1
		self.tracker_service.registerTarget(self.targetName, self.faceWidth)

		self.port = 6666
		hostname=host
		ip = self.get_ip_address('wlan0') 
		IPAddr=socket.gethostbyname(hostname)		
		print("Your Computer IP Address is:"+ip)
		address = (ip, self.port) 
		self.socket = None
		self.listener = Listener(address)

	def get_ip_address(self,ifname):
		s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
		return socket.inet_ntoa(fcntl.ioctl(s.fileno(),0x8915,struct.pack('256s', ifname[:15]))[20:24])



	def connect(self,listener):

		socket = listener.accept()
		print('Connection accepted from', self.listener.last_accepted)
		return socket

	def exit(self):
		self.tablet_service.hideImage()
		self.camProxy_service.unsubscribe(self.upper_cam)
		sys.exit(0)


	def image_to_byte_array(self,image):
		imgByteArr = io.BytesIO()
		image.save(imgByteArr, 'png')
		imgByteArr = imgByteArr.getvalue()
		return imgByteArr

	def value_confidence(self,value,name):
		text = ""
		if(value[name]["value"] != 0):
			text += "\n==========="+name.upper()+"==========="
			text += "\nValue: \t"+str(value[name]["value"])
			text += "\nConfidence: \t"+str(value[name]["confidence"])
		return text

	def level_confidence(self,value,name1,name2):
		text = ""
		if(value[name1][name2]["level"] != 0):
			text += "\n==========="+name1.upper()+"==========="
			text += "\nValue: \t"+str(value[name1][name2]["level"])
			text += "\nConfidence: \t"+str(value[name1][name2]["confidence"])
		return text

	def values_confidence(self,value,name1,name2):
		text = ""
		if(value[name1][name2]["value"] != 0):
			text += "\n==========="+name2.upper()+"==========="
			text += "\nValue: \t"+str(value[name1][name2]["value"])
			text += "\nConfidence: \t"+str(value[name1][name2]["confidence"])
		return text

	def get_person_state(self,mood):
		person_state = mood.currentPersonState()
		valence = person_state["valence"]
		attention = person_state["attention"]
		body_language = person_state["bodyLanguageState"]
		ease = body_language["ease"]

		text = self.value_confidence(person_state,"valence")
		text += self.value_confidence(person_state,"attention")
		text += self.level_confidence(person_state,"bodyLanguageState","ease")
		text += self.value_confidence(person_state,"smile")
		text += self.values_confidence(person_state,"expressions","calm")
		text += self.values_confidence(person_state,"expressions","anger")
		text += self.values_confidence(person_state,"expressions","joy")
		text += self.values_confidence(person_state,"expressions","sorrow")
		text += self.values_confidence(person_state,"expressions","laughter")
		text += self.values_confidence(person_state,"expressions","excitement")
		text += self.values_confidence(person_state,"expressions","surprise")
		return text

	def cam(self):
		self.basic_awareness_service.startAwareness()
		self.tracker_service.track(self.targetName)
		time_start = time.time()
		image_count = 0

		images = []
		image_time_start= time.time()
		duration = 1.0/self.fps
		run_time = duration
		#self.leds_service.rasta(1)
		while(image_count<self.n_images):
			
			if(run_time>=duration):
				#self.leds_service.rasta(0.05)
				#print(str(run_time) +" "+str(time.time()-time_start))
				image_time_start= time.time()
				yimg = self.camProxy_service.getImageRemote(self.upper_cam)
				'''
				dimg = self.camProxy_service.getImageRemote(self.depth)
				image=np.zeros((dimg[1], dimg[0]),np.uint8)
				values=map(ord,list(dimg[6]))
				j=0
				for y in range (0,dimg[1]):
					for x in range (0,dimg[0]):
						image.itemset((y,x),values[j])
						j=j+1
				'''
				
				if(yimg != None):
					try:
						'''
						person_info = self.get_person_state(self.moodService)
						if(person_info != ""):
							print(person_info)
						else:
							print("No person info")
						'''



						array = yimg[6]
						imageWidth = yimg[0]
						imageHeight = yimg[1]
						#self.analyse_face(yimg)
						send_array = base64.b64encode(array)
						byte_image = json.dumps(send_array)
						images.append(byte_image)


						#im.show()

					except  Exception as e: 
						exc_type, exc_obj, exc_tb = sys.exc_info()
						fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
						print(exc_type, fname, exc_tb.tb_lineno)
						print(e)
						self.exit()
				else:

					print("Warning: None Image. Creating empty image.")
					image = Image.new('RGB', self.imageWidth, self.imageHeight)
					image_bytes = self.image_to_byte_array(image)
					images.append(image_bytes)

				image_count += 1
			run_time = time.time()-image_time_start
		time_end = time.time()		

		print("Acquisition delay ", time_end - time_start)	
		self.basic_awareness_service.stopAwareness()
		self.tracker_service.stopTracker()

		#images = pickle.dumps(images)
		return images

	def set_image_lar(self):
		self.tablet_service.showImage("http://198.18.0.1/img/lar.png")

	def set_image_error(self):
		self.tablet_service.showImage("http://198.18.0.1/img/larNotConnected.png")

	def set_image_get_states(self):
		self.tablet_service.showImage("http://198.18.0.1/img/larGetStates.png")

	def set_image_action(self):
		self.tablet_service.showImage("http://198.18.0.1/img/larAction.png")



	def distance(self,pt_1, pt_2):
		pt_1 = np.array((pt_1[0], pt_1[1]))
		pt_2 = np.array((pt_2[0], pt_2[1]))
		return np.linalg.norm(pt_1-pt_2)


	def analyse_face(self,yimg):
		array = yimg[6]
		imageWidth = yimg[0]
		imageHeight = yimg[1]
		#array = bytearray(array)
		image = Image.frombytes("L", (imageWidth, imageHeight), str(bytearray(array)))
		frame = np.array(image)
		#frame = frame[:, :, ::-1].copy()

		is_to_crop = True
		crop_img = frame
		if is_to_crop:
			face_cascade = cv.CascadeClassifier('/usr/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
			#gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
			gray = frame
			#cv.imshow("image",frame)
			#cv.waitKey(0)
			#cv.destroyAllWindows()

			faces = face_cascade.detectMultiScale(gray, 1.1, 10)
			#img_h, img_w, channels = frame.shape
			#centers of image
			center_img = [imageWidth/2,imageHeight/2]
			#search more centralized face
			nearest_coord = []
			print('Faces: '+str(len(faces)))
			if len(faces) > 0:
				print('Nfaces '+str(faces))
				nearest_coord = faces[0]
				min_distance = self.distance(center_img,nearest_coord[:2])

				#(x,y,w,h)
				for f in faces:
					#cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
					#roi_gray = gray[y:y+h, x:x+w]
					#roi_color = img[y:y+h, x:x+w]
					#eyes = eye_cascade.detectMultiScale(roi_gray)
					#for (ex,ey,ew,eh) in eyes:
					#	cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,25x:x+w5,0),2)
					(x,y,w,h) = f
					aux_distance = self.distance(center_img,f[:2])
					if aux_distance < min_distance:
						min_distance = aux_distance
						nearest_coord = f

				(x,y,w,h) = nearest_coord
				crop_img = frame[y:y+h, x:x+w]
		fx = 0.55
		fy = 0.55
		small_frame = cv.resize(frame, (0, 0), fx=fx, fy=fy)

		# Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
		#rgb_small_frame = small_frame[:, :, ::-1]
		crop_img = small_frame

		directory = os.path.dirname(os.path.realpath(__file__))
		#pickle_image = pickle.dumps(crop_img)







	def verify_substring(self,sub,data):
		return data.replace(sub,'').replace(' ','').replace(':','')


	#Then specify the resolution among : kQQVGA (160x120), kQVGA (320x240),
	#kVGA (640x480) or k4VGA (1280x960, only with the HD camera).
	def set_cam_resolution(self,resolution_name='kQVGA'):
		self.camProxy_service.unsubscribe(self.upper_cam)
		resolution = vd.kQVGA
		if(resolution_name == 'kQQVGA'):
				resolution = vd.kQQVGA
		elif(resolution_name == 'kQVGA'):
				resolution = vd.kQVGA
		elif(resolution_name == 'kVGA'):
				resolution = vd.kVGA
		elif(resolution_name =='k4VGA'):
				resolution = vd.k4VGA
		self.upper_cam = self.camProxy_service.subscribeCamera("Ucam",0, resolution, self.colorSpace, 20)
		return self.upper_cam




	def proccess_command(self,data):
		#Configure rewards
		if 'reward neutral' in data:
			result = self.verify_substring('reward neutral',data)
			self.neutral_reward = float(result.replace(',','.'))
			return "0"
		elif 'reward hs_success' in data:
			result = self.verify_substring('reward hs_success',data)
			self.hs_success_reward = float(result.replace(',','.'))
			return "0"
		elif 'reward hs_fail' in data:
			result = self.verify_substring('reward hs_fail',data)
			self.hs_fail_reward = float(result.replace(',','.'))
			return "0"
		elif 'reward eg_fail' in data:
			result = self.verify_substring('reward eg_fail',data)
			self.eg_fail_reward = float(result.replace(',','.'))
			return "0"
		elif 'reward eg_success' in data:
			result = self.verify_substring('reward eg_success',data)
			self.eg_success_reward = float(result.replace(',','.'))
			return "0"

		elif 'resolution' in data:
			result = self.verify_substring('resolution',data)
			self.set_cam_resolution(result)
			return "0"
		#Manage States
		elif 'get_screen' in data:
			self.set_image_get_states()
			images_to_send = self.cam()			
			return images_to_send
		elif 'close_socket' in data:
			self.connected = False

		elif data.isdigit():
			self.set_image_action()
			reward = self.execute(int(data))
			print("Sending reward: "+str(reward))
			return "reward "+str(reward)
		else:
			print("Unknown data: "+str(data))
			return "1"

			



	def execute(self,action):
		print("Executing: "+str(action))
		reward = 0
		if action == 1:
			reward = self.wait()
		else:
			self.basic_awareness_service.startAwareness()
			self.tracker_service.track(self.targetName)
			#self.tracker_service.start_new_thread(cam,(step,num2,))
			if action == 2:
				time.sleep(1)
			elif action == 3:
				reward = self.hello()
				time.sleep(2)
			elif action == 4:
				reward = self.shake_hand()
			self.basic_awareness_service.stopAwareness()
			self.tracker_service.stopTracker()
			
		return reward



	def onTouched(self, strVarName, value):
		""" This will be called each time a touch
		is detected.

		"""
		# Disconnect to the event when talking,
		# to avoid repetitions
		self.touch.signal.disconnect(self.id)
		touched_bodies = []
		for p in value:
			if p[0] in self.dict_sensors:
				self.dict_sensors[p[0]] = p[1]	
		# Reconnect again to the event
		self.id = self.touch.signal.connect(functools.partial(self.onTouched, "TouchChanged"))


	def rightHandSensor(self):
		if self.dict_sensors['RArm']:
			return 1
		else:
			return 0


	def wait(self):
		names =['HeadYaw','HeadPitch']
		times=[[0.7],[0.7]]
		time_sleep = 1
		opt = randrange(7)
		async = True

		if opt==1:
			#print 'I am in 1'
			self.motion_service.angleInterpolation(names,[0.0,-0.16],times,async)

			self.motion_service.setAngles(names,[0.0,-0.26179],0.2)
			
		elif opt==2:
			#print 'I am in 2'
			self.motion_service.angleInterpolation(names,[0.2,-0.1],times,async)

			self.motion_service.setAngles(names,[0.0,-0.26179],0.2)
		elif opt==3:
			#print 'I am in 3'
			self.motion_service.angleInterpolation(names,[0.2,-0.1],times,async)
			
			self.motion_service.setAngles(names,[0.0,-0.26179],0.2)
		elif opt==4:
			#print 'I am in 4'
			self.motion_service.angleInterpolation(names,[-0.4,-0.1],times,async)

			self.motion_service.setAngles(names,[0.0,-0.26179],0.2)
		elif opt==5:
			#print 'I am in 5'
			self.motion_service.angleInterpolation(names,[0.0,-0.26179],times,async)

			self.motion_service.setAngles(names,[0.0,-0.26179],0.2)
		elif opt==6:
			#print 'I am in 6'
			self.motion_service.angleInterpolation(names,[0.0,-0.26179],times,async)

			self.motion_service.setAngles(names,[0.0,-0.26179],0.2)
		time.sleep(time_sleep)
		#print('Returning')
		return 0			
		

	def hello(self):
		names = list()
		times = list()
		keys = list()

		names.append("LElbowRoll")
		times.append([1, 1.5, 2, 2.5])
		keys.append([-1.02102, -0.537561, -1.02102, -0.537561])

		names.append("LElbowYaw")
		times.append([1, 2.5])
		keys.append([-0.66497, -0.66497])

		names.append("LHand")
		times.append([2.5])
		keys.append([0.66])

		names.append("LShoulderPitch")
		times.append([1, 2.5])
		keys.append([-0.707571, -0.707571])

		names.append("LShoulderRoll")
		times.append([1, 2.5])
		keys.append([0.558505, 0.558505])

		names.append("LWristYaw")
		times.append([1, 2.5])
		keys.append([-0.0191986, -0.0191986])
		names2=["LElbowRoll","LElbowYaw","LHand","LShoulderPitch","LShoulderRoll","LWristYaw"]
		angles=[-0.479966,-0.561996,0.66,1.30202,0.195477, -0.637045]
		self.motion_service.setExternalCollisionProtectionEnabled("Arms", False)
		self.tts_service.setParameter("speed", 100)
		self.tts_service.setLanguage("English")
		self.motion_service.angleInterpolation(names, keys, times, True)
		self.tts_service.say("Hello")
		self.motion_service.setAngles(names2,angles,0.3)
		
		return 0
			
		

	def shake_hand(self):
		names = list()
		times = list()
		keys = list()
		keys_shake = list()
		times_shake = list()
		r=0
		names.append("RHand")
		times.append([2])
		keys.append([0.98])

		times_shake.append([0.25])
		keys_shake.append([0.98])



		names.append("RShoulderPitch")
		times.append([2])
		keys.append([-0.2058])

		times_shake.append([0.25])
		keys_shake.append([0.2058])
		
		
		names2=["RElbowRoll","RElbowYaw","RHand","RShoulderPitch","RShoulderRoll","RWristYaw"]
		angles=[0.479966,0.561996,0.66,1.30202,-0.195477, 0.637045]
		
		names3=["RHand"]
		angles2=[0.5]
		
		self.motion_service.setExternalCollisionProtectionEnabled("Arms", False)

		self.tts_service.setParameter("speed", 60)
		self.tts_service.setLanguage("English")
		self.motion_service.setExternalCollisionProtectionEnabled("Arms", False)
		#self.motion_service.angleInterpolation(names, keys, times, True)
		
		self.tts_service.say("My name is Pepper!")
		time_start = time.time()
		time_update = time.time()
		while(time_update-time_start<5):
			self.motion_service.angleInterpolation(names, keys, times, True)
			r=self.rightHandSensor()
			if int(r)>0:
				break
			time_update = time.time()

		if int(r)>0:
			self.tts_service.say("Nice to meet you")
			#thread.start_new_thread(touch_sensor,(str(2),))
			self.motion_service.setExternalCollisionProtectionEnabled("Arms", False)

			self.motion_service.setAngles(names3,angles2,0.6)
			for i in range(2):
				self.motion_service.angleInterpolation(names, keys_shake, times_shake, True)
				#time.sleep(0.5)
				self.motion_service.angleInterpolation(names, keys, times_shake, True)

		self.motion_service.setAngles(names2,angles,0.1)
		return r



	def send(self,data):
		self.socket.send(json.dumps(data))
		print("Sended")

	def receive(self):
		print("Waiting")
		message = self.socket.recv()
		message = json.loads(message)
		print("Received: "+str(message))
		return message

	def run(self):
		try:
			while(True):
				if self.socket != None:
					try:
						self.set_image_lar()
						operation = self.receive()
						print("Getting: "+str(operation))
						response = self.proccess_command(operation)
						self.send(response)
						self.set_image_lar()
					except IOError as e:
						print("Except Message: "+str(e))
						print("Socket connection closed.")
						self.socket = None
					except EOFError as e:

						print("Error: "+str(e))
						self.socket = None
				else:
					self.set_image_error()
					print("Wating for new connection...")
					self.socket = self.connect(self.listener)


		except KeyboardInterrupt as e:
			print("Exiting: "+str(e))
			self.exit()
			



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--ip", type=str, default="pepper.local",
						help="Robot IP address. On robot or Local Naoqi: use '127.0.0.1'.")
	parser.add_argument("--port", type=int, default=9559,
						help="Naoqi port number")
	parser.add_argument("--socket_host", type=str, default="pepper.local",
						help="Socket Host")

	args = parser.parse_args()
	try:
		# Initialize qi framework.
		connection_url = "tcp://" + args.ip + ":" + str(args.port)
		app = qi.Application(["PepperHRI", "--qi-url=" + connection_url])
	except RuntimeError:
		print ("Can't connect to Naoqi at ip \"" + args.ip + "\" on port " + str(args.port) +".\n"
			   "Please check your script arguments. Run with -h option for help.")
		sys.exit(1)
	app.start()
	session = app.session
	host = args.socket_host
	pepperHRI = PepperHRI(session,host)
	pepperHRI.run()



