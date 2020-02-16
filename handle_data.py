import simplejson as json # will be needed to work with our json files

def load_data(filename):
    # loads the data, currently is implemented to handle json input
    print("Loading the file: " + filename + "...\n")
    with open(filename, 'r') as jfile:
        data = json.load(jfile) # decodes the json file into a list
    return data

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

def extract_sentence_pairs(data): # extract sentence pairs
    pairs = [] # final data structure to be returned. list of sentence pairs
    for d in data: # for each dictionary in the list (each convo)
        for i in range(0, len(d['dialog'])): # for each dictionary in the dialog key (list)
            if i < (len(d['dialog']) - 1):  # if less than the last elm
                  pairs.append((d['dialog'][i]['text'], d['dialog'][i+1]['text']))
    return pairs

def print_sentence_pairs(pairs): # prints the extracted sentence pairs
    for i in pairs:
        print("0 -> " + str(i[0]) + "\n1 -> " + str(i[1]) + "\n")
        
def handle_data(filename): # puts all of our functions together in handling data
    print("\nLoading Data\n")
    data = load_data(filename) # first, load our json data into our process 
    #print_json_data(data) # custom function written to actually be able to read the json file lol
    print("Reformatting Data into Sentence Pairs\n")
    pairs = extract_sentence_pairs(data) # returns a list of sentence pairs stripped from each conversation
    print("Sentence Pairs Extracted\n")
    #print_sentence_pairs(pairs)


#convert json data to txt data
'''
    def writeDataIntoTxt(data):
    # utterance data file
    f1 = open("./data/utterance.txt", "w")
    # conversation data file
    f2 = open("./data/conversation.txt", "w")
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
            #print(utteranceData)
            f1.write(utteranceData)
        # get rid of the last ","
        dialogUtterances = dialogUtterances[:-1] + "]"
        conversationData = sender + "+++$+++" + receiver + "+++$+++" + dialogUtterances + " \n"
        f2.write(conversationData)
    f1.close()
    f2.close()
'''
