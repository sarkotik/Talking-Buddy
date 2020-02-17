# Default word tokens and their respective indexes in our dictionaries
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token

class Voc:
    def __init__(self, name):
        self.name = name # name of thie vocab
        self.trimmed = False # keeps track in if this vocabulary class has been trimmed
        self.wordToIndex = {} # stores each word (key) and its index (value) in a dictionary
        self.wordToCount = {} # stores each word (key) and its count (value) in a dictionary
        self.indexToWord = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"} # stores each index (index) and its word (value) as a dictionary
        self.num_words = 3  #  starts at 3 b/c SOS, EOS, PAD are already in it // will be used as the index of each word in wordToIndex

    def addSentence(self, sentence): # adds by sentence into our vocab by calling addWord() on each word 
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word): # adds word to this vocabulary class
        if word not in self.wordToIndex: # word is not already documented by our class
            self.wordToIndex[word] = self.num_words # set the word's (key) of the wordToIndex dictionary to our index tracker's count
            self.wordToCount[word] = 1 # adding new word, so count is 1
            self.indexToWord[self.num_words] = word # set the index (key) of the indexToWord dictionary to our word // is the reverse of 2 lines up
            self.num_words += 1 # increase our index tracker by 1 because we are adding one more word to our class
        else: # word is already documented by our class
            self.wordToCount[word] += 1 # increase the tracked count of this word by 1

    def trim(self, min_count): # remove words that are below a certain count threshold (their wordToCount value)
        if self.trimmed: # already trimmed
            return
        self.trimmed = True # the vocab is now trimmed after this function

        # now iterate through each key and check its count in wordToCount dictionary
        for k, v in self.wordToCount.items():
            if v < min_count:
                print("POPPED 1") 
                self.wordToCount.pop(k) # pop from word count dictionary
                self.indexToWord.pop(self.wordToIndex.pop(k)) # second pop pops by word and returns index to first pop which pops by index
                self.num_words -= 1 # decrement index tracker by 1
                
        
            
            
