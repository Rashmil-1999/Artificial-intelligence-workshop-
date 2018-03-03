# -*- coding: utf-8 -*-
'''
face detection using haar cascade
'''
import cv2

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier("../trained_models/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("../trained_models/haarcascade_eye.xml")

#image = cv2.imread("../test_images/IMG-20130817-01388.jpg")
#image = cv2.resize(image, (720,600))
while True:
    ret_val, frame = cap.read()
    if ret_val == True:
        
        
        faces = face_cascade.detectMultiScale(frame, 1.1, 8)# image source, scale parameter to zoom, neighbour conditions that need to be satisfied for it to classify as a face
        
        for (x,y,w,h) in faces:
            im=frame[y:y+h,x:x+w]
            cv2.rectangle(frame, (x,y), (x+w,y+h),(0,0,255),3)
            eye = eye_cascade.detectMultiScale(im, 1.1, 10)
            for (x1,y1,w1,h1) in eye:
                cv2.rectangle(im, (x1,y1), (x1+w1,y1+h1),(0,0,255),3)
        
        cv2.imshow("face_detect", frame)
        key_pressed = cv2.waitKey(1) & 0xFF
        if key_pressed == 27:
            break
cv2.destroyAllWindows()
cap.release()    