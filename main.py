import time
import math
import os # used to access files, more specifically, to delete  mp3 files in our case
import sys # will be needed for sys.exit() to quit program whenever we want
from signal import signal, SIGINT # needed to catch control-c, SIGINT termination

# our other scripts, in the same directory
import speechToText as STT # imports speechToText.py, which handles the speech to text input
import textToSpeech as TTS # imports textToSpeech.py, which handles the text to speec output
import rnn as RNN # imports our rnn.py script to be used as our trained model
import handle_data as HD # imports all of our custom  functions written to handle the json data file for training

def sigint_handler(signal, frame): # implements the sigint handler to terminate gracefully
    print("\n-={ SIGINT (CONTROL-C) caught }=-")
    sys.exit(0)

def loop_convo(): # handles the input, prediction, and output
    # start the conversation loop, quit when user says "quit conversation"
    file_number = 1 # counts and makes files discernable by number
    while(1):
        # first, capture input from user
        input_text = STT.capture_input() # this method from speechToText.py returns the captured voice input as text
        print("Input: "  + input_text)
    
        if(input_text == "quit conversation"):
            sys.exit(0) # quits the program if user says "quit conversation"

        # second, predict a response to the input query and transform to verbal form for output

        # third, execute the verbal output
        TTS.text_to_speech(input_text, file_number) # calls our custom text to speech function/script

        # increment file number (mp3 file name as a number)
        file_number+=1

def clean_up(): # deletes all .mp3 files in this directory
    for file in os.listdir("./sounds_data/"):
        if file.endswith(".mp3"):
            try:
                os.remove("./sounds_data/" + file)
            except:
                print("removal didnt work")
    
if __name__ == '__main__':
    clean_up() # clean up all saved mp3s (outputs) at beginning, so this run's recordings are saved until next run
    
    print("Welcome to Talking Buddy!")
    TTS.text_to_speech("Welcome to Talking Buddy!", "start")
    
    signal(SIGINT, sigint_handler)# sets up sigint handler for graceful terrmination
    
    pairs = HD.handle_data("./data/data.json") # load the data with the data file in the same directory as this main.py
    # pairs is the list of all pairs of sentences from the conversations


    # now, build the vocabulary using our Vocabulary class in our vocab.py module
    

 #   loop_convo() # loop conversation with user // quits the program when user says "quit conversation"



   
