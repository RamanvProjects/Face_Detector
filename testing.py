from FaceDetector import FaceDetector
import cv2

a = cv2.imread('testimg.jpg')
c = FaceDetector(a)

d = c.get_faces()

cv2.imshow('img',d[1])
cv2.waitKey(0)
cv2.destroyAllWindows()
