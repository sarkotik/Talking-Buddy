import simplejson as json # will be needed to work with our json files
import vocab  # our custom module/class to build our vocabulary with
import re # regular expressions will be used for simple string replacements

### -------- LOAD DATA -------- ### 
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

### -------- EXTRACT SENTENCE PAIRS FROM DATA -------- ### 
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

### -------- PREPARE DATA AND BUILD VOCAB --------  ###
MAX_LENGTH = 10 # maximum sentence length to be considered. can be changed

def stringToAscii(s): # takes a string and converts each char to ascii
    return ''.join(
        c for c ini unicodedata.normalize('NFD', s)
        if unicodedata.catagory(c) != 'Mn'
    )

def normalizeString(s)  # takes a string and lowercases, trims, and removes non-letter chars
    # calls our stringToAscii function in order to convert to ascii
    s = stringToAscii(s.lower().strip())  # lowercases and strips white spaces on the left and right of the string
    s = re.sub(r"([.!?])", r" \1", s)  # gets rid of punctuation
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s) # only allows these kind of chars
    s = re.sub(r"\s+", r" ", s).strip() # strips remaining white spaces
    return s

### -------- HANDLE DATA MAIN -------- ### 
def handle_data(filename): # puts all of our functions together in handling data
    print("\nLoading Data\n")
    data = load_data(filename) # first, load our json data into our process
    #print_json_data(data) # custom function written to actually be able to read the json file lol

    print("Reformatting Data into Sentence Pairs\n")
    pairs = extract_sentence_pairs(data) # returns a list of sentence pairs stripped from each conversation
    #print_sentence_pairs(pairs)

    print("Preparing Data and Building Vocabulary")
    # now that we have the pairs, build the vocab and alter pairs to fit what we want


