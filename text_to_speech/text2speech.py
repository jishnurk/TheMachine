#!/usr/bin/python

from text_to_speech import *


def tts(typ,text):
    if typ == 't':
        print text
        speech_eng.say(text)
        speech_eng.runAndWait()
    elif get_type(typ) != 0:
        i = random.randrange(len(get_type(typ)))
        print get_type(typ)[i]+text
        speech_eng.say(get_type(typ)[i]+text)
        speech_eng.runAndWait()

def get_type(typ):
    return {
        'g': greet,
        'pr': pos_resp,
        'nr': neg_resp,
    }.get(typ,0)
