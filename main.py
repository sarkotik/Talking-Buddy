import time
import math
import os # for os.walk() and such to deal with our directory's files
import sys # will be needed for sys.exit() to quit program whenever we want
import simplejson as json # will be needed to work with our json files


# our other scripts, in the same directory
import speechToText as STT # imports speechToText.py, which handles the speech to text input
import textToSpeech as TTS # imports textToSpeech.py, which handles the text to speec output
import rnn # imports our rnn.py script to be used as our trained model

def load_data(filename):
    print("Loading the file: " + filename + "...\n")
    with open(filename, 'r') as jfile:
        data = json.load(jfile) # decodes the json file into a list

    print('\nTotal number of dialogues extracted =', len(data))
    


def loop_convo():
    # start the conversation loop, quit when user says "quit conversation"
    while(1):
        input_text = STT.capture_input() # this method from speechToText.py returns the captured voice input as text
        print("Input: "  + input_text)
        
        if(input_text == "quit conversation"):
            sys.exit() # quits the program if user says "quit conversation"


if __name__ == '__main__':
    print("Welcome to Talking Buddy!")
    load_data("./data.json")
    loop_convo() # loop conversation with user // quits the program when user says "quit conversation"
    



   
