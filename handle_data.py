import simplejson as json # will be needed to work with our json files
from vocab import Voc  # our custom module/class to build our vocabulary with
import re # regular expressions will be used for simple string replacements
import unicodedata  # needed to convert strings to ascii

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

def stringToAscii(s): # takes a string and converts each char to ascii, called by normalizeString
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):  # takes a string and lowercases, trims, and removes non-letter chars
    # calls our stringToAscii function in order to convert to ascii
    s = stringToAscii(s.lower().strip())  # lowercases and strips white spaces on the left and right of the string
    s = re.sub(r"([.!?])", r" \1", s)  # gets rid of punctuation
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s) # only allows these kind of chars
    s = re.sub(r"\s+", r" ", s).strip() # strips remaining white spaces
    return s

def testPairLengths(pair): # takes in a tuple (query/response pair) and returns true iff both sentences are under the MAX_LENGTH threshold
    return len(pair[0].split(' ')) <= MAX_LENGTH and len(pair[1].split(' ')) <= MAX_LENGTH  # currently <=10

def filterPairs(pairs): # takes in the data object (list of tuple pairs) and filters pairs by max length by using testPairLengths
    return [pair for pair in pairs if testPairLengths(pair)]

def buildVocab(pairs, corpus_name): # takes in query/response pairs and return a voc object. also filters the list pairs and returns new list of pairs
    # data is currently a list of tuple pairs of query/response pairs
    voc = Voc(corpus_name) # instantiate our Vocab object with a name
    pairs = filterPairs(pairs) #  call our custom filterPairs() on our pairs to filter by max length
    for pair in pairs: # add each sentence of each pair into vocab by addSentence()
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])
    return voc, pairs # want to return our build vocab and our filtered pairs

def trimRareWords(voc, pairs, MIN_COUNT): # takes in a Voc object and a pairs list and remove words that have a count < min_count
    voc.trim(MIN_COUNT) # call our voc object's trim method on itself
    keep_pairs = []
    for pair in pairs: # check each sentence in each pair to see if they contain rare words
        keep_0 = True
        keep_1 = True
        for word in pair[0].split(' '):
            if word not in voc.wordToIndex:
                keep_0 = False
                break
        for word in pair[1].split(' '):
            if word not in voc.wordToIndex:
                keep_1 = False
                break
        if keep_0 and keep_1:
            keep_pairs.append(pair)
    return keep_pairs
        
### -------- HANDLE DATA MAIN -------- ### 
def handle_data(filename): # puts all of our functions together in handling data
    print("\nLoading Data...\n")
    # first, load our json data into our process
    data = load_data(filename)
    #print_json_data(data) # custom function written to actually be able to read the json file lol

    print("Reformatting Data into Sentence Pairs...\n")
    # extract sentence pairs from data and normalize them
    pairs = extract_sentence_pairs(data) # returns a list of sentence pairs stripped from each conversation
    #print_sentence_pairs(pairs)
    pairs = [[normalizeString(s) for s in pair] for pair in pairs] # normalize all strings by lowercasing, trimming, and removing non-letter chars

    print("Preparing Data and Building Vocabulary...\n")
    # now that we have the pairs, build the vocab and refine the pairs list by removing pairs with sentences > max_words
    voc, pairs = buildVocab(pairs, "Talking Buddy Corpus") # returns a built vocabulary from training data as well a refined (by max len) list of pairs
    
    #print(voc.wordToCount)
    #print(pairs)
    print("number of pairs: " + str(len(pairs)))
    print("number of unique words: " + str(voc.num_words))

    # lastly, we want to refine our pairs list once more by removing pairs that contain rare words
    pairs = trimRareWords(voc, pairs, 2) # min count set to 2
    
    # return voc and pairs
    return voc, pairs
