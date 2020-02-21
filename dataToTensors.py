# contains functions to be used to convert our data to tensors to be fed into our rnns
import itertools

import vocab


# change the word in the sentence to id
def indexesFromSentence(voc, sentence):
    return [voc.indexToWord[word] for word in sentence.split(' ')] + [vocab.EOS_token]


"""
padding the list with PAD_token.
@l: the list of lists
@return: the list of lists which is after padding. the size is (len(l), the max length of lists)
"""
def zeroPadding(l, fillValue=vocab.PAD_token):
    return list(itertools.zip_longest(*l, fillValue=fillValue))

"""
convert the value in the lists of list to binary
if the value is not padding, it should be 1. if the value is padding, it should be 0.
@l: the list of lists
@return: return a binary matrix
"""
def binaryMatrix(l, padding = vocab.PAD_token):
    # the return matrix
    binaryMatrix = []
    # iterate the l
    for i in range(len(l)):
        # create a new row in the binary matrix
        binaryMatrix[i] = []
        # iterate the No.i element in l
        for token in l[i]:
            # the value is padding, assign 0
            if token == vocab.PAD_token:
                binaryMatrix[i].append(0)
            # the value is not padding, assign 1
            else:
                binaryMatrix[i].append(1)
    return binaryMatrix
