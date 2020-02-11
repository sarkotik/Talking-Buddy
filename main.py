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
    # loads the data, currently is implemented to handle json input
    print("Loading the file: " + filename + "...\n")
    with open(filename, 'r') as jfile:
        data = json.load(jfile) # decodes the json file into a list

    print_json_data(data) # custom function written to actually be able to read the json file lol
        
def print_json_data(data): # type of data is a list of dictionaries
    print('\nTotal number of dialogues extracted =', len(data))
    for d in data: # iterate through the outer list, each d is a dictionary
        for i in d: # iterate through all of the keys ofeach dictionary d
            if i == "dialog": # special prints out the dialog key of the dictionary, most essential part
                print(str(i) + "  ->  " )
                for v in d[i]: # iterates through all of the dialogues in dialogue (each v is a dictionary and dialogue is a list of dictionaries) 
                      print(v)
                print()
            else: # prints out all of the other keys and their values
                print(str(i) + "  ->  " + str(d[i]) + "\n\n")
        
        print("-------------------------------------------------------------------------------------------------------------------\n-------------------------------------------------------------------------------------------------------------------\n")

def loop_convo():
    # start the conversation loop, quit when user says "quit conversation"
    while(1):
        # first, capture input from user
        input_text = STT.capture_input() # this method from speechToText.py returns the captured voice input as text
        print("Input: "  + input_text)
    
        if(input_text == "quit conversation"):
            sys.exit() # quits the program if user says "quit conversation"

        # second, predict a response to the input query and transform to verbal form for output


if __name__ == '__main__':
    print("Welcome to Talking Buddy!")
    load_data("./data.json") # load the data with the data file in the same directory as this main.py
    #loop_convo() # loop conversation with user // quits the program when user says "quit conversation"
    



   
