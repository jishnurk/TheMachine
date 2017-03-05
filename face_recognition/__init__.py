#!/usr/bin/python

import cv2, Image, os, numpy, pyttsx, random
from threading import Thread


################### Variables used in Face_Recognition Module ###################

faces_path = 'face_recognition/faces/'
xml_path = 'face_recognition/xml/'
i=0
labels = numpy.empty(0)
first_run = True
conf = 0
nbr_predicted = 0
neg_resp = ["Sorry, I am still learning.", "Pardon me, I don't know you."]

face_cascade = cv2.CascadeClassifier(xml_path+"haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(xml_path+"haarcascade_eye.xml")
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
recognizer = cv2.createLBPHFaceRecognizer()

speech_eng = pyttsx.init()
