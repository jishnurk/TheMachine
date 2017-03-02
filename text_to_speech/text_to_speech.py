#!/usr/bin/python

import pyttsx, random

greet = ["Hi ", "Hello "]
pos_resp = ["My pleasure."]
neg_resp = ["Sorry, I am learning.", "Pardon me, I don't know you."]

def tts(typ,text):
    speech_eng = pyttsx.init()
    if get_type(typ) != 0:
        speech_eng.say(get_type(typ)[random.randrange(len(get_type(typ)))]+text)
        speech_eng.runAndWait()

def get_type(typ):
    return {
        'g': greet,
        'pr': pos_resp,
        'nr': neg_resp,
    }.get(typ,0)
