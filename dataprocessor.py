from FaceDetector import FaceDetector
from os import listdir
import cv2

imgs = [cv2.imread('./data/' + x) for x in listdir('./data')]

c = FaceDetector()

for img in imgs:
    c.update_img(img)
    poses, fdetect_img = c.get_faces()
    
    x, y, w, h = poses[0][0]
    cv2.imshow('frame', img[y:y+h, x:x+w])
    cv2.waitKey()
print imgs

