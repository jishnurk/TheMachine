#!/usr/bin/python

import cv2, Image, numpy as np, signal, sys
from face_recognition import face_recog
from text_to_speech import text_to_speech
from threading import Thread

run = True


def update():
    while True:
        global cap, ret, img
        if not run:
            cap.release()
            cv2.destroyAllWindows()
            return
        ret, img = cap.read()


def face_recog_module():
    while True:
        global img, run
        try:
            out_img, name = face_recog.recog_face(img)
        except:
            print "Error in Face Recognition module! : ", sys.exc_info()[0]
            run = False
            return
        cv2.imshow('Webcam 0',out_img)
        k = cv2.waitKey(50)
        if k == 27: # Exit if ESC is pressed
            run = False
            return


print "################## the Machine v_alpha_1 ##################\n########### Jishnu Radhakrishnan : Manoj S Nair ###########\n\n"
cap = cv2.VideoCapture(0)
ret, img = cap.read()
Thread(target=face_recog_module, args=()).start()
Thread(target=update, args=()).start()

#text_to_speech.tts('g',name)
#thread.start_new_thread(speech_eng.runAndWait, ())
