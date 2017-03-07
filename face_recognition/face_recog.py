#!/usr/bin/python

from face_recognition import *


def load_recognizer_data():
    if [f for f in os.listdir(xml_path) if f.startswith("face_recog_data")] :
        global labels, recognizer
        recognizer.load(xml_path+"face_recog_data.xml")
        #print "Recognizer Data loaded."
        labels = numpy.array(numpy.loadtxt(xml_path+"/face_data.xml", unpack=True, dtype=str, ndmin=1))
    else:
        raise Exception("Face Recognizer is untrained. Train it first!")
    return    


def update_recognizer(face,name):
    global labels, recognizer
    labels = numpy.append(labels,name)
    recognizer.update(numpy.array([cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)]), numpy.array(len(labels)-1))
    #print "Recognizer updated."
    recognizer.save(xml_path+"face_recog_data.xml")
    f = open(xml_path+"face_data.xml", "w")
    numpy.savetxt(f, numpy.column_stack([labels]), fmt='%s')
    #print "Saved Recognizer Data to xml\\face_recog_data.xml and xml\\face_data.xml"
    return

def get_faces(img):
    gray = clahe.apply(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    return face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(75, 75), maxSize=(500, 500))

def get_eyes(img):
    gray = clahe.apply(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    return eye_cascade.detectMultiScale(gray, minSize=(img.shape[1]/5, img.shape[1]/5), maxSize=(img.shape[1]/3, img.shape[1]/3))

def recog_face(img):
    global nbr_predicted, conf
    faces = []
    ids = []
    load_recognizer_data()
    img_shw = numpy.array(img)
    for (x,y,w,h) in get_faces(img):
        faces = numpy.append(faces, [x,y,w,h])        
        nbr_predicted, conf = recognizer.predict(clahe.apply(cv2.cvtColor(img[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)))
        if conf<90:            
            ids = numpy.append(ids, labels[nbr_predicted])
        else:
            ids = numpy.append(ids, "unknown")
    return faces, ids
