#!/usr/bin/python

import numpy as np
import cv2, Image, os, pyttsx
import face_crop
import thread

#################### Variables ####################

faces_path = 'faces'
xml_path = 'xml'
debug = False
i=0
old_name=""
new_name=""
eye_marks=[]
greetings = ["Hi", "Hello"]
neg_resp = ["Sorry, I am learning", "Pardon me, Who are you?"]
images = []
labels = []


#################### FUNCTIONS ####################

def get_images_and_labels(path):
    image_paths = [os.path.join(faces_path, f) for f in os.listdir(faces_path)]
    for image_path in image_paths:
        image_pil = Image.open(image_path).convert('L')
        image = np.array(image_pil, 'uint8')
        image = cv2.equalizeHist(image)
        nbr = os.path.split(image_path)[1].split("_")[0]
        images.append(image)
        labels.append(nbr)
    return images, labels

def load_recognizer_data():
    if [f for f in os.listdir(xml_path) if f.startswith("face_recog_data")] :
        recognizer.load("xml/face_recog_data.xml")
        print "Loaded Recognizer Data."
        images, labels = get_images_and_labels(faces_path)
    else:    
        train_recognizer()

def train_recognizer():
    images, labels = get_images_and_labels(faces_path)
    print "Faces added."
    if images:
        recognizer.train(images, np.array(range(0,len(labels))))
        print "Recognizer trained."
        recognizer.save("xml/face_recog_data.xml")
        print "Saved Recognizer Data to xml/face_recog_data.xml."        
    return

def update_recognizer(face,name):
    recognizer.update(np.array(face), [name])
    print "Recognizer updated."
    recognizer.save("xml/face_recog_data.xml")
    print "Saved Recognizer Data to xml/face_recog_data.xml."        
    return


#################### Start ####################

print "########### the Machine v0.1 ##########\n########### Jishnu Radhakrishnan ##########\n\n"
inp = input("Start Face Recognition? Y/N : ")
if inp=="Y":
    face_cascade = cv2.CascadeClassifier('xml/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('xml/haarcascade_eye.xml')
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    recognizer = cv2.createLBPHFaceRecognizer()
    speech_eng = pyttsx.init()

    load_recognizer_data()

    cap = cv2.VideoCapture(0)

    print "Face Recognition started."
    
    while True:
        ret,img = cap.read()
        img_shw = np.array(img)
        gray = clahe.apply(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(75, 75), maxSize=(500, 500))
        for (x,y,w,h) in faces:            
            face_gray = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(face_gray, minSize=(w/5, w/5), maxSize=(w/3, w/3))
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(img_shw[y:y+h, x:x+w],(ex,ey),(ex+ew,ey+eh),(255,0,0),2)
            #cropd = face_crop.CropFace(img, eye_left=(eye_marks[0],eye_marks[1]), eye_right=(eye_marks[2],eye_marks[3]), offset_pct=(0.2,0.2))
            #cv2.imshow('Face',cropd) 
            nbr_predicted, conf = recognizer.predict(face_gray)
            print "Log : Predicted "+labels[nbr_predicted]+" with confidence %d" %(conf)                
            if conf>90:
                cv2.rectangle(img_shw,(x,y),(x+w,y+h),(0,0,255),2)
                cv2.imshow('Webcam 0',img_shw)                    
                speech_eng.say("Sorry, I don't know you. Please type your name")
                print "Sorry, I don't know you. Please type your name"
                thread.start_new_thread( speech_eng.runAndWait, () )
                user = input("Your Name : ")                
                if user == '' :
                    user="anon"                    
                #Get path of all faces captured and set value of 'i' accordingly to avoid overwriting
                temp_path = [os.path.join(faces_path, f) for f in os.listdir(faces_path) if f.startswith(user)]
                if temp_path:
                    i = int(os.path.split(temp_path[-1])[1].split(".")[0].replace(user+"_","")) + 1
                file = "faces/"+user+"_%i.jpg" %(i)
                if debug:
                    print "Writing image to : "+file
                print "Face captured."
                cv2.imwrite(file, img[y:y+h, x:x+w])                
                update_recognizer(img[y:y+h, x:x+w], user)
            else:
                cv2.rectangle(img_shw,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.putText(img_shw, labels[nbr_predicted], (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(255,255,255),2)
                speech_eng.say("Hi "+labels[nbr_predicted])            
        cv2.imshow('Webcam 0',img_shw)    
        k = cv2.waitKey(50)# Exit if ESC is pressed
        if k == 27:
            break
        elif k == 100: # Key 'd'
            debug = not debug
            print "Debug : %i" %(debug)           
    cap.release()
    cv2.destroyAllWindows()
