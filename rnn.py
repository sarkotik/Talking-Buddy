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
        self.embedding = embedding # used to create embeddings from input words
        self.hidden_size = hidden_size
        
        # initialize our GRU. input_size is set to hidden_size because our input size is a word embedding with # of features == hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout = (0 if n_layers == 1 else dropout), bidirectional = True)

    # side note - defining a forward method for a NN allows us to call backward() to compute gradients
    def forward(self, input_seq, input_lengths, hidden = None):
        # step 1 -> convert word indexes to embeddings
        embedded =  self.embedding(input_seq)
        
        # step 2 -> pack padded batch of sequences to be fed into this RNN by calling pack_padded_sequence()
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)

        # step 3 -> forward pass this packed data through the defined GRU
        outputs, hidden = self.gru(packed, hidden)  # gru returns outputs and hidden state to be fed into the next time step

        # step 4 -> unpack padding by calling pad_packed_sequence
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)

        # step 5 -> sum together the bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]

        # return the final output and hidden state
        return outputs, hidden


# our Luong attention layer -> implements "Global Attention"
# read https://blog.floydhub.com/attention-mechanism/#luong-att for more info on attention layers used to prevent information loss
# Luong's global attention takes the last context vector and concatenates them with the last output vector as an input to RNN
# the current context vector will be passed to the next time step
# Differences from Bahdanau's Attention Mechanism -> need all of the encoder's hidden states and just the hidden state of the decoder from the current time step only
class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()

        # three different ways that the alignment scoring function is defined -> dot, general, and concat
        self.method = method
        if self.method not in ['dot', 'general', 'concat']: # error check for the three allowed methods
            raise ValueError(self.method, "is not one of the 3 allowed methods (dot, general, concat)")
        self.hidden_size = hidden_size
        if self.method == 'general': # general method case
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat': # concat method case
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output): # simplest of the three functions 
        # produce the alignment score by multiplying the hidden states of the encoder and the hidden state of the decoder
        # Alignment_Score = H_e *  H_d
        return torch.sum(hidden * encoder_output, dim = 2)

    def general_score(self, hidden, encoder_output): # similar to dot function, except a weight matrix is added into the equation
        # Alignment_Score = W(H_e *  H_d)
        attn_weights = self.attn(encoder_output)
        return torch.sum(hidden * attn_weights, dim = 2)
        
    def concat_score(self, hidden, encoder_output): # similar to the way alignment scores are calculated in Bahdanau's Attention Mechanism
        # the decoder hidden state is added to the encoder's hidden states
        # Alignment_Score = W * tanh(W_combined(H_e + H_d))
        attn_weights = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * attn_weights, dim = 2)
    
    # defining forward allows us to call backward to compute gradients of a calculation
    def forward(self, hidden, encoder_outputs):
        # calculate the attention weights based on the given method (3 possible)
        if self.method == "general": # general method
            attn_weights = self.general_score(hidden, encoder_outputs)
        elif self.method == "concat": # concat method
            attn_weights = self.concat_score(hidden, encoder_outputs)
        elif self.method == "dot": # dot method
            attn_weights = self.dot_score(hidden, encoder_outputs)
        else: # extra safeguard
            raise ValueError(self.method, "is not one of the 3 allowed methods (dot, general, concat)")
        
        # transpose the max_length and batch_size dimensions
        attn_weights = attn_weights.t() # flip matrix across the line y = -x (\)

        # return the result, normalized with a softmax function in order for probability conversions (0-1)
        return F.softmax(attn_weightts,  dim = 1).unsqueeze(1)

        
# our decoderRNN model which implements our attention layer class above
# forward pass will be through a unidirectional GRU, in comparison to our encoder rnn's biidirectional GRU's
class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers = 1, dropout = 0.1):
        super(LuongAttnDecoderRNN,   self).__init__()

        # initialize our decoder rnn's attributes to its parameters upon initialization
        self.attn_model = attn_model # will be one of three things -> dot, concat, and general
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # define our decoder rnn's layers
        self.embedding = embedding  # used to embed input words
        self.embedding_dropout = nn.Dropout(dropout)
        # initialize our GRU. input_size is set to hidden_size because our input size is a word embedding with # of features == hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout = (o if n_layers == 1 else dropout)) 
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size) # ouput layer 

        self.attn = Attn(attn_model, hidden_size) # creates an instance of our attention layer class above

    # we will run this one step (word) at a time
    def forward(self, input_step, last_hidden, encoder_outputs):
        # step 1 -> get embedding of the current input word
        embedded = self.embedding(input_step)
        embedded = self.embedding_drouput(embedded)

        # step 2 -> forward pass through unidirectional GRU, get final output and hidden state from GRU
        rnn_output, hidden = self.gru(embedded, last_hidden)

        # step 3 -> calculate the attention weights from the current GRU output
        # In order to calculate attention weights, Luong's Global Attention uses the hidden state of the decoder from the current time step only
        attn_weights = self.attn(rnn_output, encoder_outputs) # calculates attn weights with the current decoder hidden state and 

        # step 4 -> multiple the attention weights to encoder outputs in order to get a new "weighted sum" context vector
        context_vector = attn_weights.bmm(encoder_outputs.transpose(0, 1))

        # step 5 -> concatenate weighted context vector and GRU output using Luong eq 5
        rnn_output = rnn_output.squeeze(0)
        context_vector = context_vector.squeeze(1)
        concat_input = torch.cat((rnn_output, context) , 1)
        concat_output = torch.tanh(self.concat(concat_input))

        # step 6 -> predict the next word in the sequence using Luong eq 6
        output = self.out(concat_output)
        output = F.softmax(output, dim=1) # use softmax function in order to normalize values to probabilistic distribution (0-1)

        # finally, return the output and the final hidden state
        return output, hidden
        

        
# our single training function
# we set the max length to 11 because we are chilling with inputs of <=10 chars
def train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, embedding, encoder_optimizer, decoder_optimizer, batch_size, clip, max_length = 11):
    # zero gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # set device options
    input_variable = input_variable.to(device)
    lengths = lengths.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)

    # NOT DONE YET ...

# evaluate function
# evaluate the response of chatbot
def evaluate(encoder, decoder, searcher, voc, sentence, max_length=handle_data.MAX_LENGTH):
    # change a batch of input sentence to index
    indexs_batch = [dataToTensors.indexsFormSentence(voc, sentence)]
    # create lengths tensor
    lengths = torch.tensor(len(indexs) for indexs in indexs_batch)
    # transpose the dimensions
    input_batch = torch.LongTensor(indexs_batch).transpose(0,1)
    # use approriate device
    input_batch = indexs_batch.to(device)
    lengths = lengths.to(device)
    # decode sentence by searcher
    tokens, score = searcher(indexs_batch, lengths, max_length)
    # change index to word
    decoded_words = [voc.indexToWord[token.item()] for token in tokens]
    return decoded_words


# predict function


