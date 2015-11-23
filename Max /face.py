import numpy as np
import cv2
import os 
from os.path import isfile, join

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# face_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
# face_cascade = cv2.CascadeClassifier('lbpcascade_profileface.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

pictures_path = "/media/maxsong/friends/vid1_1fps"
files = os.listdir(pictures_path)
pictures = [f for f in files if f[-4:] == ".png"]

#sorted(pictures)
#sorted(pictures, key=lambda x: len(x))
#print pictures
for i, file_name in enumerate(pictures):
    if True: 
	file_name_minus_ext = file_name[:-4]
	file_path = join(pictures_path,file_name)
	img = cv2.imread(file_path)
	img = cv2.resize(img, (0,0), fx=1, fy=1) 
	# img = cv2.resize(img, (0,0), fx=4, fy=4) 
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	
	faces = face_cascade.detectMultiScale(gray, 1.3, 2)
	for (x,y,w,h) in faces:
	    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
	roi_gray = gray[y:y+h, x:x+w]
	roi_color = img[y:y+h, x:x+w]
	eyes = eye_cascade.detectMultiScale(roi_gray)
	for (ex,ey,ew,eh) in eyes:
	    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
	
	img = cv2.resize(img, (0,0), fx=0.5, fy=0.5) 
	#cv2.imshow('img',img)
	cv2.imwrite( "processed_images/{}_processed.png".format(file_name_minus_ext), img);
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
