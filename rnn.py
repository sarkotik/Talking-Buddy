##will handle train, predict, our nn model, etc
from io import open
import os
import random
import time
import math
import torch.nn as nn
import torchvision as tv

# our RNN model
class RNN(nn.Module):
    def __init__(self, input_size, hideden_size, output_size):
        super(RNN, self).__init__()

        # declaration of the RNN's attributes
        self.hidden_size = hidden_size # sets up hidden size arg as a self attribute
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size) # goes to hidden and back to original hidden
        self.i2o = nn.Linear(input_size + hidden_size, output_size) # goes to softmax and then output
        self.softmax = nn.LogSoftmax(dim=1) # sets up softmax layer to get a score  between  0 an 1 for probability reasonings

    def forward(self, input, hidden):
        # Put the computation for forward pass here
        combined = torch.cat((input, hidden), 1) # sets up the combined layer of input and hidden, before i2o or i2h
                
        hidden = self.i2h(combined) # sends combined through i2h, going to be sent back to hidden
        output = self.i2o(combined) # sends combined through i20, producing output
        output = self.softmax(output) # takes output and passes through softmax before final return
        return output, hidden
            
    def initHidden(self):
        return torch.zeros(1, self.hidden_size)







# train function




# predict function


