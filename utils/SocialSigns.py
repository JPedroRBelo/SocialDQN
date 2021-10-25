import sys
sys.path.append('../')
import numpy as np
import cv2
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

class SocialSigns:
	def __init__(self,):
		self.scaleFactor = 1.2
		self.minNeighbors = 1
		self.minSize = (5,5)


	def find_faces(self,image):
		
		gray = np.array(image)

		scale_percent = 200 # percent of original size
		width = int(gray.shape[1] * scale_percent / 100)
		height = int(gray.shape[0] * scale_percent / 100)
		dim = (width, height)

		# resize image
		gray = cv2.resize(gray, dim, interpolation = cv2.INTER_AREA)
		#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
		faces = faceCascade.detectMultiScale(
		    gray,
		    scaleFactor=self.scaleFactor,
		    minNeighbors=self.minNeighbors,
		    minSize=self.minSize
		)
		show_image = False
		if(show_image):
			for (x, y, w, h) in faces:
				cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 2)
				roi_color = gray[y:y + h, x:x + w]
				#cv2.imwrite(str(w) + str(h) + '_faces.jpg', roi_color)

			cv2.imshow('Gray image', gray)		  
			cv2.waitKey(1)		
		



		return len(faces)

	'''
	def main(self):
		path = '../images/1/gray'

		eps = 0

		max_index = 8

		file_exists = True
		while file_exists:
			face_found = False

			for i in range(max_index):
				file_name = path+str(eps)+"_"+str(i)+".png"
				image = Path(file_name)
				if image.is_file():
					file_exists = True
					image = cv2.imread(file_name,1)
					face = self.find_face(image)
					if(face):
						face_found = True
						
				else:
					file_exists = False
			if(not face_found):
				print("[F] No face detected on ep"+ str(eps))
			else:
				print("[T] Face detected on ep"+ str(eps))
					
			eps += 1

	'''

