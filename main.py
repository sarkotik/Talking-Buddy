import time
import math
import os # used to access files, more specifically, to delete  mp3 files in our case
import sys # will be needed for sys.exit() to quit program whenever we want
from signal import signal, SIGINT # needed to catch control-c, SIGINT termination
import torch
import torch.nn as nn
import torchvision as tv

# our other scripts, in the same directory
import rnn_models as RNN # imports our rnn.py script to be used as our trained model
import handle_data as HD # imports all of our custom  functions written to handle the json data file for training
import train_and_evaluate as TAE # imports all of our training and evaluation functions
import text_to_speech as TTS # imports textToSpeech.py, which handles the text to speec output

# USE_CUDA and device are global variables that are availably accessed by each module where they are instantiated as global
USE_CUDA = torch.cuda.is_available() # if cuda is available to run on machine's GPU, use it
device = torch.device("cuda" if USE_CUDA else "cpu") # define the device we will use to run our Pytorch stuff

# global variable save_dir in order to set where we save our checkpoint model after tranining
save_dir = os.path.join("data", "save")

def sigint_handler(signal, frame): # implements the sigint handler to terminate gracefully
    print("\n-={ SIGINT (CONTROL-C) caught }=-")
    sys.exit(0)

def loop_convo(encoder, decoder, voc): # handles the input, prediction, and output
    # start the conversation loop, quit when user says "quit conversation"
    file_number = 1 # counts and makes files discernable by number
    # Set dropout layers to eval mode
    encoder.eval()
    decoder.eval()
    # Initialize search module
    searcher = TAE.GreedySearchDecoder(encoder, decoder)
    # loop the conversation with evaluateInput() from our TAE module
    TAE.evaluateInput(encoder, decoder, searcher, voc)
    # print to acknowledge quit, once exiting from evaluateInput()'s loop
    print("Conversation has ended")

def call_train(encoder, decoder, model_name, voc, pairs, loadFilename, embedding, encoder_n_layers, decoder_n_layers, batch_size, hidden_size): # case where we want to train our models from scratch, calls trainIters() and train() from TAE script
    # configure our training and optimization
    clip = 50.0
    teacher_forcing_ratio = 1.0 # determines if we want to use teacher forcing when we are calling our train() function
    learning_rate = 0.0001
    decoder_learning_ratio = 5.0
    n_iteration = 1000 # we want to iterate 1000 times, originally 4000
    print_every = 1  # print every iteration
    save_every = 250 # save every 250 iterations, originally 500

    # ennsure that our dropout layers are in training mode
    encoder.train()
    decoder.train()

    # initialize optimizers
    print("Building Optimizers...")
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr = learning_rate)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr = learning_rate * decoder_learning_ratio)

    if loadFilename: # case where we are loading the model instead of training one from scratch
        encoder_optimizer.load_state_dict(encoder_optimizer_sd)
        decoder_optimizer.load_state_dict(decoder_optimizer_sd)

    # if you have cuda, configure cuda to call in order to run this on the GPU instead of the CPU
    for state in encoder_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor): # if the value v in state is of type torch.Tensor
                state[k] = v.cuda()

    for state in decoder_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

    # run the actual training iterations
    print("Starting the training iterations...")
    TAE.trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer, embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size, print_every, save_every, clip, model_name, loadFilename, teacher_forcing_ratio, hidden_size)    

def clean_up(): # deletes all .mp3 files in this directory
    for file in os.listdir("./sounds_data/"):
        if file.endswith(".mp3"):
            try:
                os.remove("./sounds_data/" + file)
            except:
                print("removal didnt work")
    
if __name__ == '__main__':
    clean_up() # clean up all saved mp3s (outputs) at beginning, so this run's recordings are saved until next run

    print("Welcome to Talking Buddy!")
    TTS.text_to_speech("Welcome to Talking Buddy!", "start")
    
    signal(SIGINT, sigint_handler) # sets up sigint handler for graceful terrmination

    voc, pairs = HD.handle_data("./training_data/data.json") # load the data with the data file in the same directory as this main.py
    # now have built vocab and refined list of pairs # pairs is ready to be converted to tensors and used
    
    #print(pairs)
    #print(voc.wordToCount)
    print("\nnumber of pairs (AFTER TRIM): " + str(len(pairs)))
    print("number of unique words (AFTER TRIM): " + str(voc.num_words) + '\n')

    # have to initialize the indeividual encoder and decoder models 
    model_name = 'Talking_Buddy'
    attn_model = 'dot' # can set to 'dot', 'general', or 'concat'
    hidden_size = 500 
    encoder_n_layers = 2
    decoder_n_layers = 2
    dropout = 0.1
    batch_size = 64

    checkpoint_iter = 1000 # we iterated 1000 times last time, pick up @1000

    # now, will have two paths -> if we want to (re)train model or load a trained model
    # set checkpoint to load from, set to None if we want to train from scratch
    #loadFilename = None # none to train, filename to load
    loadFilename = os.path.join(save_dir, model_name, model_name, '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size), '{}_checkpoint.tar'.format(checkpoint_iter))

 

    # load model if loadFilename is not None
    # Load model if a loadFilename is provided
    if loadFilename:
        print("Loading Model....")
        # If loading on same machine the model was trained on
        checkpoint = torch.load(loadFilename)
        # If loading a model trained on GPU to CPU
        #checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
        # load checkpoints from the previously trained, saved model
        encoder_sd = checkpoint['en']
        decoder_sd = checkpoint['de']
        encoder_optimizer_sd = checkpoint['en_opt']
        decoder_optimizer_sd = checkpoint['de_opt']
        embedding_sd = checkpoint['embedding']
        voc.__dict__ = checkpoint['voc_dict']

    print('Building encoder and decoder ...\n')

    # Initialize word embeddings
    embedding = nn.Embedding(voc.num_words, hidden_size)
    if loadFilename:
        embedding.load_state_dict(embedding_sd)

    # Initialize encoder & decoder models
    encoder = RNN.EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
    decoder = RNN.LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)

    if loadFilename: # case where we are loading trained models of our encoder and decoder 
        encoder.load_state_dict(encoder_sd)
        decoder.load_state_dict(decoder_sd)
    else: # case where we want to train our models from scratch
        call_train(encoder, decoder, model_name, voc, pairs, loadFilename, embedding, encoder_n_layers, decoder_n_layers, batch_size, hidden_size) # case where we want to train our models from scratch, calls trainIters() and train() from TAE script) # call our call_train() functions in order to call tranIters() and train() in order to train our models
        
    # Use appropriate device
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    print('Models built and ready to go!')

    # loop convo with our evaluateInput() function in our TAE script
    loop_convo(encoder, decoder, voc) # loop conversation with user // quits the program when user says "quit conversation"



   
