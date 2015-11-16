import numpy as np
import cv2


class FaceDetector():
    def __init__(self, img=None):
        basepath = '/Users/ramanv/Documents/sublime/gits/opencv/data/haarcascades/'
        self.face_cascade = cv2.CascadeClassifier(basepath + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(basepath + 'haarcascade_eye.xml')
        self.img = img
        self.faceDetectedImg = None
        self.positions = None

    def update_img(self, img):
        self.img = img
        self.faceDetectedImg = None
        return True

    def get_img(self):
        return self.img

    def get_faces(self):
        if self.faceDetectedImg is not None:
            return (self.positions, self.faceDetectedImg)

        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        self.faceDetectedImg = self.img

        poses = []

        for (x,y,w,h) in faces:
            cv2.rectangle(self.faceDetectedImg,(x,y),(x+w,y+h),(255,0,0),2)

            roi_gray = gray[y:y+h, x:x+w]
            roi_color = self.faceDetectedImg[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(roi_gray)

            pos = ((x,y,w,h),[])
            for (ex,ey,ew,eh) in eyes:
                pos[1].append((ex,ey,ew,eh))
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            poses.append(pos)

        self.positions = poses
        return (self.positions, self.faceDetectedImg)
