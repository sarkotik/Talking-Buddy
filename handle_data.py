import simplejson as json # will be needed to work with our json files

def load_data(filename):
    # loads the data, currently is implemented to handle json input
    print("Loading the file: " + filename + "...\n")
    with open(filename, 'r') as jfile:
        data = json.load(jfile) # decodes the json file into a list

    print_json_data(data) # custom function written to actually be able to read the json file lol
    writeDataIntoTxt(data)
        
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

#convert json data to txt data
def writeDataIntoTxt(data):
    # utterance data file
    f1 = open("./utteracne.txt", "w")
    # conversation data file
    f2 = open("./conversation.txt", "w")
    # id counter
    counter = 0
    # utterance list
    utterances = []
    for d in data:
        # the list of every utterances in a dialog
        dialogUtterances = "["
        # get participant1 id
        participant1 = d.get("participant1_id").get("user_id")
        #get participant2 id
        participant2 = d.get("participant1_id").get("user_id")
        dialog = d.get("dialog")
        # assign sender's and receiver's id
        sender = participant1 if d.get("sender") == participant1 else participant2
        receiver = participant2 if d.get("sender") == participant1 else participant1
        for v in dialog:
            # find whether the utterance exist
            if v.get("text") in utterances:
                index = utterances.index(v.get("text"))
            else:
            # not exist create a new id
                utterances.append(v.get("text"))
                index = counter
                counter += 1
            dialogUtterances = dialogUtterances + str(index) + ","
            utteranceData = str(index) + "+++$+++" + sender + "+++$+++" + v.get("text") + " \n"
            print(utteranceData)
            f1.write(utteranceData)
        # get rid of the last ","
        dialogUtterances = dialogUtterances[:-1] + "]"
        conversationData = sender + "+++$+++" + receiver + "+++$+++" + dialogUtterances + " \n"
        f2.write(conversationData)
    f1.close()
    f2.close()





