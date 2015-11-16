import numpy as np
import cv2
from FaceDetector import FaceDetector

cap = cv2.VideoCapture(0)

c = FaceDetector()
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    if ret:
        c.update_img(frame)
        d = c.get_faces()
    else:
        d = (0, frame)

    cv2.imshow('frame',d[1])
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
