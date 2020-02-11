import time
import math
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
    
if __name__ == '__main__':
    print("Welcome to Talking Buddy!")
    signal(SIGINT, sigint_handler)
    HD.load_data("./data.json") # load the data with the data file in the same directory as this main.py
    loop_convo() # loop conversation with user // quits the program when user says "quit conversation"
    



   
