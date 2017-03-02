#!/usr/bin/python

import numpy as np
import cv2, Image, os, pyttsx
import random

neg_resp = ["Sorry, I am learning.", "Pardon me, I don't know you."]

#################### Variables ####################

faces_path = 'face_recognition/faces/'
xml_path = 'face_recognition/xml/'
i=0
labels = []
first_run = True
conf = 0
nbr_predicted = 0

#################### FUNCTIONS ####################

def load_recognizer_data():
    if [f for f in os.listdir(xml_path) if f.startswith("face_recog_data")] :
        global first_run, labels, recognizer
        first_run = False
        recognizer.load(xml_path+"face_recog_data.xml")
        #print "Recognizer Data loaded."
        labels = np.array(np.loadtxt(xml_path+"/face_data.xml", unpack=True, dtype=str, ndmin=1))                    
        

def update_recognizer(face,name):
    global first_run, labels, recognizer
    labels.append(name)
    recognizer.update(np.array([cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)]), np.array(len(labels)-1))
    #print "Recognizer updated."
    recognizer.save(xml_path+"face_recog_data.xml")
    f = open(xml_path+"face_data.xml", "w")
    np.savetxt(f, np.column_stack([labels]), fmt='%s')
    #print "Saved Recognizer Data to xml\\face_recog_data.xml and xml\\face_data.xml"    
    first_run = False
    return


#################### Start ####################

def recog_face(img):
    global recognizer, nbr_predicted, conf, i
    face_cascade = cv2.CascadeClassifier(xml_path+"haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier(xml_path+"haarcascade_eye.xml")
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    recognizer = cv2.createLBPHFaceRecognizer()

    speech_eng = pyttsx.init()

    load_recognizer_data()

    img_shw = np.array(img)   
    gray = clahe.apply(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(75, 75), maxSize=(500, 500))
    for (x,y,w,h) in faces:
        face_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(face_gray, minSize=(w/5, w/5), maxSize=(w/3, w/3))
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(img_shw[y:y+h, x:x+w],(ex,ey),(ex+ew,ey+eh),(255,0,0),2)
        if(first_run == False):
            nbr_predicted, conf = recognizer.predict(face_gray)
            print "Predicted "+labels[nbr_predicted]+" with confidence %d" %(conf)                
        if conf>90 or (first_run == True):
            cv2.rectangle(img_shw,(x,y),(x+w,y+h),(0,0,255),2)
            cv2.imshow('Webcam 0',img_shw)                    
            speech_eng.say(neg_resp[random.randrange(len(neg_resp))]+"Please type your name")
            speech_eng.runAndWait()
            print "Sorry, I don't know you. Please type your name"                
            user = input("Your Name : ")
            if user == '' :
                user="anon"                    
            #Get path of all faces captured and set value of 'i' accordingly to avoid overwriting
            temp_path = [os.path.join(faces_path, f) for f in os.listdir(faces_path) if f.startswith(user)]
            if temp_path:
                i = int(os.path.split(temp_path[-1])[1].split(".")[0].replace(user+"_","")) + 1
            file = faces_path+user+"_%i.jpg" %(i)
            print "Face captured."
            cv2.imwrite(file, img[y:y+h, x:x+w])
            update_recognizer(img[y:y+h, x:x+w], user)
        else:
            cv2.rectangle(img_shw,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(img_shw, labels[nbr_predicted], (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(255,255,255),2)
    return img_shw, labels[nbr_predicted]
