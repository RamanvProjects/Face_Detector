from FaceDetector import FaceDetector
import cv2
import os

imgs = [cv2.imread('./data/' + x) for x in os.listdir('./data')]

c = FaceDetector()

if not os.path.exists('./processed'):
    os.mkdir('./processed')

num = 0
for img in imgs:
    c.update_img(img)
    poses, fdetect_img = c.get_faces()
    
    x, y, w, h = poses[0][0]
    num+=1

    face = img[y:y+h, x:x+w]

    # processing face (eigenface dealio)
    
# Insert processing here
    cv2.imwrite('./processed/' + str(num) + '.jpg', face);

