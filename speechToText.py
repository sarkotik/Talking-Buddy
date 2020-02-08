import random
import time

# using 2.7.15 and 3.6.4 versions of python works

import speech_recognition as sr

def recognize(recognizer, mic):
    if not isinstance(recognizer, sr.Recognizer): # catches TypeError for Recognizer
        raise TypeError("wrong instance 1")
    if not isinstance(mic, sr.Microphone): # catches TypeError for Microphone
        raise TypeError("wrong instance 2")

    with mic as source: 
        recognizer.adjust_for_ambient_noise(source) # get rid of constant background nose frequencies
        audio = recognizer.listen(source) 

    response = {
        "success": True, # True if this func succeeds, else False
        "error": None,  # stores error data, None if no error, else store error msg 
        "transcription": None  # store voice input as text, we want this at the end
    }

    try: # try to capture voice input
        response["transcription"] = recognizer.recognize_google(audio)
    except sr.RequestError: # catch api error
        response["success"] = False
        response["error"] = "API unavailable"
    except sr.UnknownValueError: # catch unrecognizable voice input
        response["error"] = "Unable to recognize speech"
    #else: # print success msg, can comment this out. leave it in for testing purposes
        #print("capture successful")
        
    return response


def capture_input():
    #create recognizer and mic instances
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    guess = recognize(recognizer, microphone) # gets voice input captured as text

    if guess["error"]: # error checking // prints BAD if an error occured
        print("BAD")
    else:       # returns voice input text if no error
        return guess["transcription"]
