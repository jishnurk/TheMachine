#!/usr/bin/python

import cv2, Image, numpy, sys, speech_recognition
from face_recognition import face_recog
from text_to_speech import text2speech
from threading import Thread

run = True
WIT_AI_KEY = "API_KEY"
speech_recog = speech_recognition.Recognizer()

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
        out_img = numpy.array(img)
        try:
        #if True:
            faces, names = face_recog.recog_face(out_img)            
            i = 0
            for (x,y,w,h) in face_recog.get_faces(out_img):
                if names[i] == "unknown":
                    cv2.rectangle(out_img,(x,y),(x+w,y+h),(0,0,255),2) #Red Border
                else:
                    cv2.rectangle(out_img,(x,y),(x+w,y+h),(0,255,0),2) #Green Border
                cv2.putText(out_img, names[i], (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(255,255,255),2)
                ++i
        except:
        #else:
            print "Error in Face Recognition module! : "+str(sys.exc_info()[1])
            run = False
            return
        cv2.imshow('Webcam 0',out_img)
        k = cv2.waitKey(50)
        if k == 27: # Exit if ESC is pressed
            run = False
            return
        
def listen():
    with speech_recognition.Microphone() as source:
        speech_recog.adjust_for_ambient_noise(source)
        audio = speech_recog.listen(source)
    try:
        return speech_recog.recognize_wit(audio, key=WIT_AI_KEY)
        #return speech_recog.recognize_google(audio)
        #return speech_recog.recognize_sphinx(audio)
    except speech_recognition.UnknownValueError:
        print("Could not understand audio")
    except speech_recognition.RequestError as e:
        print("Recog Error; {0}".format(e))
    return ""


print "################## the Machine v_alpha_1 ##################\n########### Jishnu Radhakrishnan : Manoj S Nair ###########\n"
text2speech.tts('t',"What do you want to do?")
text2speech.tts('t',"1. Start Facial Recognition.")
text2speech.tts('t',"2. Train Facial Recognizer.")
opt = input("Enter Choice: ")
if opt == 1:
    cap = cv2.VideoCapture(0)
    ret, img = cap.read()
    Thread(target=face_recog_module, args=()).start()
    Thread(target=update, args=()).start()
elif opt == 2:
    text2speech.tts('t',"\nYou can train the recognizer in 2 ways:")
    text2speech.tts('t',"1. Webcam\n2. Images")
    opt = input("Choose your option: ")
    if opt == 1:
        cap = cv2.VideoCapture(0)
        train_again = True 
        while(train_again):
            ret, img = cap.read()
            out_img = numpy.array(img)
            tmp_img = numpy.array(img)
            x = None
            for (x,y,w,h) in face_recog.get_faces(img):
                cv2.rectangle(out_img,(x,y),(x+w,y+h),(0,0,255),2)
                cv2.imshow('Webcam 0',out_img)
                k = cv2.waitKey(1)
                in_name = raw_input("Enter your Name ot Enter 'n' to view next Face detected: ")
                if str(in_name) == '' :
                    break
                elif str(in_name) == 'n':
                    next
                else:    
                    #Get path of all faces captured and set value of 'i' accordingly to avoid overwriting
                    #temp_path = [os.path.join(faces_path, f) for f in os.listdir(faces_path) if f.startswith(user)]
                    #if temp_path:
                    #    i = int(os.path.split(temp_path[-1])[1].split(".")[0].replace(user+"_","")) + 1
                    #file = faces_path+user+"_%i.jpg" %(i)
                    #print "Face captured."
                    #cv2.imwrite(file, img[y:y+h, x:x+w])
                    face_recog.update_recognizer(tmp_img[y:y+h, x:x+w], in_name)
            if x is None:
                print "No face detected!"
            inp = raw_input("Do you want to continue training? (Y/N): ")
            if not (str(inp)=='Y' or str(inp)=='y'):
                train_again = False
        cap.release()
        cv2.destroyAllWindows()
    elif opt == 2:
        print "This option is not yet available."
    else:
        print "Invalid choice! Run the program again."
else:
    print "Invalid choice! Run the program again."
sys.exit(0)
