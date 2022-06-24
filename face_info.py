import cv2
import numpy as np
import dlib
from time import perf_counter
import os, sys
from deepface import DeepFace
from collections import Counter


# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__



class FaceDetection:
	def __init__(self,face_detection_model="DNN",emotion_recogntion_model=""):
		self.dirr = "images/"
		self.NO_FACE = "no face"
		self.UNKNOWN_EMOTION = "unknown"
		self.CONF_NO_FACE = 0.5

		self.LABEL_HOG = "HOG"
		self.LABEL_DNN = "DNN"
		self.LABEL_MMOD = "MMOD"
		
		self.face_conf_threshold = 0.5
		self.face_detection_model = face_detection_model
		self.net = None
		self.hogDetector = None
		self.mmodDetector = None 

		self.DNN_CONV = 1000

		if(self.face_detection_model==self.LABEL_DNN):
			self.net = self.get_DNN_net()
		elif(self.face_detection_model==self.HOG):
			#Hog face detector 
			self.hogDetector = self.get_HOG_model()
		else: 
			#MMOD 
			self.mmodDetector = self.get_MMOD_model()

	def get_DNN_net(self):
		modelFile = "opencv_face_detector_uint8.pb"
		configFile = "opencv_face_detector.pbtxt"
		return cv2.dnn.readNetFromTensorflow(modelFile, configFile)

	def get_HOG_model(self):
		return  dlib.get_frontal_face_detector()

	def get_MMOD_model(self):
		return dlib.cnn_face_detection_model_v1("./mmod_human_face_detector.dat")




	def emotion_recogntion(self,frame,backend='opencv'):
		backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe']
		#image_channels = image.convert('RGB')
		#frame = np.array(image_channels)
		emotion_value = {self.UNKNOWN_EMOTION:self.CONF_NO_FACE}
		rects = []
		try:
			blockPrint()
			analyze = DeepFace.analyze(img_path = frame,actions = ['emotion'],enforce_detection= True,detector_backend = backend,prog_bar = False)
			enablePrint()
			#print(analyze['age'])
			#print(analyze['gender'])
			#print(analyze['dominant_race'])
			region = analyze['region']
			x1 = region['x']
			y1 = region['y']
			x2 = region['x']+region['w']
			y2 = region['y']+region['h']
			#image1 = cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
			#plt.imshow(image1)
			#plt.show()
			rects.append((x1,y1,x2,y2))
			emotion_value = analyze['emotion']
			#emotion_conf = analyze['emotion'][emotion]
			#emotion_value = [emotion,emotion_conf]
		except:
			enablePrint()
			emotion_value  = {self.UNKNOWN_EMOTION:self.CONF_NO_FACE}
		return emotion_value,rects



	def DNN_detection(self,frame):

		if(self.net==None):
			self.net = self.get_DNN_net()
		
		frameHeight = frame.shape[0]
		frameWidth = frame.shape[1]
		blob = cv2.dnn.blobFromImage(frame, 1.3, (self.DNN_CONV, self.DNN_CONV), [104, 117, 123], False, False)
		
		self.net.setInput(blob)
		detections = self.net.forward()
		bboxes = []
		rects = []
		for i in range(detections.shape[2]):
			confidence = detections[0, 0, i, 2]
			if confidence > self.face_conf_threshold:
				x1 = int(detections[0, 0, i, 3] * frameWidth)
				y1 = int(detections[0, 0, i, 4] * frameHeight)
				x2 = int(detections[0, 0, i, 5] * frameWidth)
				y2 = int(detections[0, 0, i, 6] * frameHeight)

				rects.append((x1,y1,x2,y2))
		return rects


	def Hog_detection(self,image):
		if(self.face_detection_model!=self.LABEL_HOG):
			self.hogDetector = self.get_HOG_model()

		frame = image.copy()
		
		faceRects = self.face_detector(frame, 0)
		frameHeight = frame.shape[0]
		frameWidth = frame.shape[1]
		rects = []
		for faceRect in faceRects:
			x1 = faceRect.left()
			y1 = faceRect.top()
			x2 = faceRect.right()
			y2 = faceRect.bottom()
			rects.append((x1,y1,x2,y2))
		return rects


	def MMOD_detection(self,image):	

		if(self.mmodDetector==None):
			self.mmodDetector = self.get_MMOD_model()
		frame = image.copy()
		
		faceRects = self.face_detector(frame, 0)
		frameHeight = frame.shape[0]
		frameWidth = frame.shape[1]
		rects = []
		for faceRect in faceRects:
			x1 = faceRect.rect.left()
			y1 = faceRect.rect.top()
			x2 = faceRect.rect.right()
			y2 = faceRect.rect.bottom()
			rects.append((x1,y1,x2,y2))
		return rects


	def choose_emotion_by_conf(self,emotions):
		dominant_emotion = self.NO_FACE
		face_visible = False
		visible_emotions = []
		for em in emotions:
			if not self.NO_FACE in em:
				visible_emotions.append(em)
		face_visible = (len(visible_emotions)>=(len(emotions)/2))

		emotion_sum = {}
		if face_visible:
			for dict_emotion in visible_emotions:
				for emotion in dict_emotion:
					if emotion in emotion_sum:
						emotion_sum[emotion] = (emotion_sum[emotion]+dict_emotion[emotion])/2
					else:
						emotion_sum[emotion] = dict_emotion[emotion]

			dominant_emotion = self.get_max_from_dict(emotion_sum)
		else:
			#print("====>"+str(len(visible_emotions)))
			dominant_emotion = self.NO_FACE

		return dominant_emotion



	def show_image(self,frame,rects,title="Image"):
		frameHeight = frame.shape[0]
		frameWidth = frame.shape[1]
		show = True
		for r in rects:
			show = True
			x1, y1, x2, y2 = r
			cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
			cv2.putText(frame,title, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 2)
		# Display
		if(show):

			cv2.imshow(title, frame)
			# Stop if escape key is pressed
			cv2.waitKey(0)

			# cv2.destroyAllWindows() simply destroys all the windows we created.
			cv2.destroyAllWindows()


	def get_max_from_dict(self,dictionary):
		fin_max = max(dictionary, key=dictionary.get)
		return fin_max




	def recognize_face_emotion(self,image,backend='mtcnn',save_path=''):
		'''
		backends = 'opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe'
		'''			

		start_time = perf_counter()
		emotion = {self.NO_FACE:self.CONF_NO_FACE}
		frame = image.convert('RGB')
		frame = np.array(frame)

		if(self.face_detection_model==self.LABEL_DNN):
			rects = self.DNN_detection(frame)
		elif(self.face_detection_model==self.LABEL_HOG):
			rects = self.Hog_detection(frame)
		else:
			rects = self.MMOD_detection(frame)
		end_time = perf_counter()
		#print(f' Face Detection Time: {end_time- start_time: 0.2f} second(s)')

		crop_img = frame			
		if(len(rects)>0):
			#Face position	
			start_time = perf_counter()	
			x1,y1,x2,y2 = rects[0]
			#Operation to crop a square image with face 
			sizeX = x2 - x1
			sizeY = y2 - y1
			square_side = max(sizeX,sizeY)*2
			meanX = int((x1+x2)/2)
			meanY = int((y1+y2)/2)
			newX1 = max(0,int(meanX-(square_side/2)))
			newX2 = min(int(meanX+(square_side/2)),frame.shape[1])
			newY1 = max(0,int(meanY-(square_side/2)))
			newY2 = min(int(meanY+(square_side/2)),frame.shape[0]) 
			#Croping image with new positions
			crop_img = frame[newY1:newY2, newX1:newX2]
			if( not (newY2<frame.shape[0] and newX2<frame.shape[1])):
				crop_img = frame				
			emotion, emotion_rects = self.emotion_recogntion(crop_img,backend=backend)
			
			
			#print(emotion)
			#self.show_image(crop_img,rects,title=self.get_max_from_dict(emotion))
		#return a tuple with emotion and confidence value

		if(save_path!=''):
			self.save_image(frame,rects,save_path=save_path,label=self.get_max_from_dict(emotion))
		return emotion

	def save_image(self,frame,rects=[],save_path="Image",label='Face'):
		frameHeight = frame.shape[0]
		frameWidth = frame.shape[1]
		show = False
		for r in rects:
			x1, y1, x2, y2 = r
			cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
			cv2.putText(frame,label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 2)

		cv2.imwrite(save_path, frame)

		

