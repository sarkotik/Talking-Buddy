# contains functions to be used to convert our data to tensors to be fed into our rnns
import itertools
import vocab
import torch


# change the word in the sentence to id
def indexesFromSentence(voc, sentence):
    return [voc.wordToindex[word] for word in sentence.split(' ')] + [vocab.EOS_token]

"""
padding the list with PAD_token.
@l: the list of lists
@return: the list of lists which is after padding. the size is (len(l), the max length of lists)
"""
def zeroPadding(l, fillvalue=vocab.PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

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
        binaryMatrix.append([])
        # iterate the No.i element in l
        for token in l[i]:
            # the value is padding, assign 0
            if token == padding:
                binaryMatrix[i].append(0)
            # the value is not padding, assign 1
            else:
                binaryMatrix[i].append(1)
    return binaryMatrix

# Returns padded input sequence tensor and lengths
def inputVar(l, voc):
    # convert the sentences to lists of ids
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    # the element in the lengths list is the length for each sentence
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # pad sentences to be the same length
    padList = zeroPadding(indexes_batch)
    # convert padList to the tensor
    padVar = torch.LongTensor(padList)
    return padVar, lengths

# Returns padded target sequence tensor, padding mask, and max target length
def outputVar(l, voc):
    # convert the sentences to lists of ids
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    # find the max length for all the sentences
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    # pad sentences to be the same length
    padList = zeroPadding(indexes_batch)
    # use binaryMatrix to determine whether the element is padded
    mask = binaryMatrix(padList)
    # use the ByteTensor to convert the element matrix to be like 1,0
    mask = torch.ByteTensor(mask)
    # convert padList to the tensor
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len

# Returns all items for a given batch of pairs
def batchToTrainData(voc, pair_batch):
    # sort the sentences according to the length of each sentence
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    # initiate the input and output list
    input_batch, output_batch = [], []
    for pair in pair_batch:
        # add first element of a pair(like a question) to the input
        input_batch.append(pair[0])
        # add second element of a pair(like a answer) to the output
        output_batch.append(pair[1])
    # process the input to be suitable for the model
    inp, lengths = inputVar(input_batch, voc)
    # process the output to be suitable for the model
    output, mask, max_target_len = outputVar(output_batch, voc)
    return inp, lengths, output, mask, max_target_len
