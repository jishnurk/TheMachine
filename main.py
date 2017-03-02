#!/usr/bin/python

import numpy as np
import cv2, Image
from face_recognition import face_recog
from text_to_speech import text_to_speech
import thread

print "################## the Machine v_alpha_1 ##################\n########### Jishnu Radhakrishnan : Manoj S Nair ###########\n\n"
cap = cv2.VideoCapture(0)

while True:
    ret,img = cap.read()
    out_img, name = face_recog.recog_face(img)
    cv2.imshow('Webcam 0',out_img)
    text_to_speech.tts('g',name)
    #thread.start_new_thread(speech_eng.runAndWait, ())
    k = cv2.waitKey(50)# Exit if ESC is pressed
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
