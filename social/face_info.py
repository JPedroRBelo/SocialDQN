import cv2
import numpy as np
import dlib
from time import perf_counter
import os, sys
from deepface import DeepFace
from collections import Counter
from fer import FER
from skimage import exposure
from skimage import img_as_float
from PIL import Image



# Disable
def blockPrint():
	sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
	sys.stdout = sys.__stdout__


class FaceDetection:
	def __init__(self):
		self.dirr = "images/"
		self.NO_FACE = "no face"
		self.UNKNOWN_EMOTION = "unknown"
		self.CONF_NO_FACE = 0.5
		self.emotion_detector = FER(mtcnn=True)


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

	def removekey(d, key):
		r = dict(d)
		del r[key]
		return r


	def normalize_img(self,image, imax=8, dtype=np.uint8):
		'''
			Normalize an image between its maximum and minimum values, and with the
			specifield caracteristics

			Params:
				image: An image to be normalized
				imax: The value of bits to represent the pixel values
				dtype: The desired dtype of the image

			Returns:
				A normalized image
		'''
		img_max = np.max(image)
		img_min = np.min(image)

		#Prevents division by 0 when the img_max and img_min have the same value
		if img_max == img_min:
			img_sub_norm = (image-img_min) / ((img_max - img_min) + 1e-12)

		else:
			img_sub_norm = (image-img_min) / (img_max - img_min)
		#Normalize image accordinly with the maximum bits representation
		#passed as parameter
		img_sub_norm = (img_sub_norm * ((2**imax) - 1)).astype(dtype)
		return img_sub_norm



	def recognize_face_emotion(self,image,preprocess=None,save_path=''):
		start_time = perf_counter()
		emotion = {self.NO_FACE:self.CONF_NO_FACE}

		frame = image.convert('RGB')
		frame = np.array(frame)
		if(preprocess != None):
			if(preprocess == 'adaptative'):
				frame = exposure.equalize_adapthist(frame, clip_limit=0.03)
				#frame = img_as_float(frame)
				#frame = np.float32(frame)
				#frame = cv2.cvtColor(image_float, cv2.COLOR_HSV2RGB)
				frame = self.normalize_img(frame)
			elif(preprocess=='stretching'):
				p2, p98 = np.percentile(frame, (2, 98))
				frame = exposure.rescale_intensity(frame, in_range=(p2, p98))
				frame = self.normalize_img(frame)
			elif(preprocess=='equalization'):
				frame = exposure.equalize_hist(frame)
				frame = self.normalize_img(frame)
				
		#up_points = (640,480)
		#frame = cv2.resize(frame, up_points, interpolation= cv2.INTER_LINEAR)
		#rects = self.DNN_detection(frame)
		#crop = self.crop_square(frame,rects)
		#norm_img = np.zeros((300, 300))
		#frame = cv2.normalize(frame, norm_img, 0, 255, cv2.NORM_MINMAX)

		#frame =  np.stack((frame,)*3, axis=-1)
		
		blockPrint()
		print('detecting')
		start_time = perf_counter()
		analysis = self.emotion_detector.detect_emotions(frame)

		enablePrint()
		rects = []
		if(len(analysis)>0):
			emotion = analysis[0]['emotions']
			rects = analysis[0]['box']
			rects[2] += rects[0]
			rects[3] += rects[1]

		if(save_path!=''):
			self.save_image(frame,rects,save_path=save_path,label=self.get_max_from_dict(emotion))

		return emotion

	def crop_square(self,frame,rects):
		crop_img = frame
		if(len(rects)>0):
			#Face position	
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
			resize_size = 200
			if( not (newY2<frame.shape[0] and newX2<frame.shape[1])):
				crop_img = frame

			elif(not (newY2<resize_size)):
				up_points = (resize_size,resize_size)
				#crop_img = cv2.resize(crop_img, up_points, interpolation= cv2.INTER_LINEAR)
				#crop_img = cv2.detailEnhance(crop_img, sigma_s=10, sigma_r=0.15)
		return crop_img

	def save_image(self,frame,rects=[],save_path="Image",label='Face'):
		frameHeight = frame.shape[0]
		frameWidth = frame.shape[1]
		show = False

		#x1, y1, x2, y2 = r
		if(len(rects)>0):
			x1 = rects[0]
			x2 = rects[1]
			y1 = rects[2]
			y2 = rects[3]
			cv2.rectangle(frame, (x1, x2), (y1, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
			cv2.putText(frame,label, (x1, x2-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 2)


		cv2.imwrite(save_path, frame)

		

