# will contain our encoder rnn and decoder rnn classes + train and predict functions
from io import open
import os
import random
import time
import math
import torch.nn as nn
import torchvision as tv

# our encoder RNN model
class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers = 1, dropout = 0):
        super(EncoderRNN, self).__init__()
        # initialize our encoder rnn's attributes to its parameters upon initialization
        self.n_layers = n_layers
        self.embedding = embedding
        self.hidden_size = hidden_size
        
        # initialize our GRU. input_size is set to hidden_size because our input size is a word embedding with # of features == hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout = (0 if n_layers == 1 else dropout), bidirectional = True)

    # side note - defining a forward method for a NN allows us to call backward() to compute gradients
    def forward(self, input_seq, input_lengths, hidden = None):
        # first, convert word indexes to embeddings
        embedded =  self.embedding(input_seq)
        
        # pack padded batch of sequences to be fed into this RNN by calling pack_padded_sequence()
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)

        # now, forward pass this packed data through the defined GRU
        outputs, hidden = self.gru(packed, hidden)  # gru returns outputs and hidden state to be fed into the next time step

        # unpack padding by calling pad_packed_sequence
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)

        # sum together the bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]

        # return the final output and hidden state
        return outputs, hidden
        
            

# our decoderRNN model



# train function




# predict function


