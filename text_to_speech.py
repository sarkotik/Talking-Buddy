# will handle our chatbot's text to speech output

from gtts import gTTS # this module is needed for text to speech (Google Text to Speech API)
import os # needed to access our system

def text_to_speech(output, number):  # takes the output of the model as text and outputs vocally
    # sets the language we want to convert to English
    language = 'en' 

    # creates a speech object by passing in the output text, langauge param, and slow param 
    speechObj = gTTS(text = output, lang = language, slow = False)

    # Saving the converted audio in a mp3 file (numbered). have to save in order to play back 
    speechObj.save("./sounds_data/" + str(number) + ".mp3") 
  
    # Playing the converted file 
    os.system("mpg123 " + "./sounds_data/" + str(number) + ".mp3")

    # Remove the converted file after outputting.
    # commented out bc was slowing down the runtime so decided to remove all mp3's outside of conversation loop
    #os.remove(str(number) + ".mp3")

    

