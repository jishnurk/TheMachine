#!/usr/bin/python

from face_recognition import *


#################### Sub Functions ####################

def load_recognizer_data():
    if [f for f in os.listdir(xml_path) if f.startswith("face_recog_data")] :
        global first_run, labels, recognizer
        first_run = False
        recognizer.load(xml_path+"face_recog_data.xml")
        #print "Recognizer Data loaded."
        labels = numpy.array(numpy.loadtxt(xml_path+"/face_data.xml", unpack=True, dtype=str, ndmin=1))
    return    


def update_recognizer(face,name):
    global first_run, labels, recognizer
    labels = numpy.append(labels,name)
    recognizer.update(numpy.array([cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)]), numpy.array(len(labels)-1))
    #print "Recognizer updated."
    recognizer.save(xml_path+"face_recog_data.xml")
    f = open(xml_path+"face_data.xml", "w")
    numpy.savetxt(f, numpy.column_stack([labels]), fmt='%s')
    #print "Saved Recognizer Data to xml\\face_recog_data.xml and xml\\face_data.xml"
    first_run = False
    return


#################### Main Function ####################

def recog_face(img):
    global nbr_predicted, conf
    load_recognizer_data()
    img_shw = numpy.array(img)
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
            speech_eng.say(neg_resp[random.randrange(len(neg_resp))]+" Please type your name.")
            print neg_resp[random.randrange(len(neg_resp))]+" Please type your name."  
            Thread(target=speech_eng.runAndWait, args=()).start()
            user = input("Your Name : ")
            if user == '' :
                user="anon"
            #Get path of all faces captured and set value of 'i' accordingly to avoid overwriting
            #temp_path = [os.path.join(faces_path, f) for f in os.listdir(faces_path) if f.startswith(user)]
            #if temp_path:
            #    i = int(os.path.split(temp_path[-1])[1].split(".")[0].replace(user+"_","")) + 1
            #file = faces_path+user+"_%i.jpg" %(i)
            #print "Face captured."
            #cv2.imwrite(file, img[y:y+h, x:x+w])
            update_recognizer(img[y:y+h, x:x+w], user)
        else:
            cv2.rectangle(img_shw,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(img_shw, labels[nbr_predicted], (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(255,255,255),2)
            return img_shw, labels[nbr_predicted]
    return img_shw, ''
