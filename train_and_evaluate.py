# contains our maskNLLLoss(), train(), and trainIters() functions to be used for training our rnn models
# also contains our GreedySearchDecoder() searcher function, evaluate(), and evaluteInput() to be used to produce output sequences to input queries based off of our trained RNN models
import os
import random
import torch
import torch.nn as nn
import torchvision as tv
import data_to_tensors
import handle_data  # imports our handle_data.py script
import speech_to_text as STT # imports speechToText.py, which handles the speech to text input
import text_to_speech as TTS # imports textToSpeech.py, which handles the text to speec output

from handle_data import normalizeString

USE_CUDA = torch.cuda.is_available() # if cuda is available to run on machine's GPU, use it
device = torch.device("cuda" if USE_CUDA else "cpu") # define the device we will use to run our Pytorch stuff

# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token

# our loss function -> calculates our loss based on our decoder's output tensor, the target tensor, and a  binary mask tensor describing the padding of the target tensor.
# -> calculates the average negative log likelihood of the elms that correspond to a 1 in the mask tensor
def maskNLLLoss(inp, target, mask):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()
        
# our single training function
# we set the max length to 11 because we are chilling with inputs of <=10 chars
def train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, embedding, encoder_optimizer, decoder_optimizer, batch_size, clip, teacher_forcing_ratio, max_length = 11):
    # zero gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # set device options
    input_variable = input_variable.to(device)
    lengths = lengths.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)

    # Initialize variables
    loss = 0
    print_losses = []
    n_totals = 0

    # forward pass thorugh the encoder
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    # create the initial decoder input (start with an SOS token for each  senttence)
    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)

    # set the initial decoder hidden state to the encoder's final hidden state
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    # decide if we are using teacher forcing for this iteration
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # forward the batch of sequences through the decoder one time step/word at a time
    if use_teacher_forcing: # the case  where teacher forcing will be used as an decoding method when training
        for t in range(max_target_len):
            # our LuongAttnDecoderRNN returns the final output and hidden state
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)

            # teacher forcing -> the next input is the current target
            decoder_input = target_variable[t].view(1, -1)
            
            # calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

    else: # not using teacher forcing, next input is the decoder's own current output
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)

            # no teacher forcing
            _, topi  = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)

            # calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

    #  perform  back propagation
    loss.backward() # by defining a forward function, we can call backward() to auto-compute the gradient of the operation on the tensor

    #  clip gradients // modified in place
    _ = nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # adjust the model weights
    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_totals


# Our trainIters() function used to handle our training iterations. calls our train() function
# runs our train() function n_iterations times
def trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer, embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size, print_every, save_every, clip, corpus_name, loadFilename, teacher_forcing_ratio, hidden_size):
    # load batches for each training iteration
    training_batches = [data_to_tensors.batchToTrainData(voc, [random.choice(pairs) for _ in range(0, batch_size)]) for _ in range(n_iteration)]

    # initializations
    print('Initializing Training Data...')
    start_iteration = 1
    print_loss = 0

    if loadFilename:  # if a filename is provided (model not yet trained)
        start_iteration = checkpoint['iteration'] + 1 # define the starting iteration for our training loop

    # Training loop
    print('Starting Training Loop...')
    for iteration in range(start_iteration, n_iteration + 1):
        training_batch = training_batches[iteration - 1]
        # extract the fields from batch
        input_variable, lengths, target_variable, mask, max_target_len = training_batch

        # run a training iteration with the batch iteration by using our single training train() function
        loss = train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, embedding, encoder_optimizer, decoder_optimizer, batch_size, clip, teacher_forcing_ratio)
        print_loss += loss

        # print the progress, we will print every iteration
        if iteration % print_every == 0:
            print_loss_avg = print_loss / print_every
            print("Iteration : {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(iteration, iteration/n_iteration * 100, print_loss_avg))
            print_loss = 0

        # save our trained model as a checkpoint in 'directory', currently set to save every 250 iterations
        if iteration % save_every == 0:
            directory = os.path.join(save_dir, model_name, corpus_name, '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size))
            if not os.path.exists(directory):  # if the directory does not already exits
                os.makedirs(directory) # create directory
            torch.save({'iteration': iteration, 'en': encoder.state_dict(), 'de': decoder.state_dict(), 'en_opt': encoder_optimizer.state_dict(), 'de_opt': decoder_optimizer.state_dict(), 'loss':  loss, 'voc_dict': voc.__dict__,  'embedding': embedding.state_dict()}, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))
        
# our searcher function, GreedySearchDecoder, to be used as our searcher in evaluate()
class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        # Initialize our encoder and decoder elms with the params
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length):
        # forward the input sequence thorugh our encoder rnn model/obj
        encoder_outputs, encoder_hidden = self.encoder(input_seq,  input_length)

        # prepare the encoder's final hidden layer to be the first hidden input to the decoder
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]

        # initialize decoder input with SOS_token
        decoder_input  = torch.ones(1, 1, device = device, dtype = torch.long) * SOS_token

        # initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device = device,  dtype = torch.long)
        all_scores =  torch.zeros([0], device = device)

        # iteratively decode one word token at a time
        for _ in range(0, max_length):
            # forward pass through our decoder rnn model/obj
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            # obtain the most likely word token and its softmaxed score
            decoder_scores, decoder_input = torch.max(decoder_output, dim = 1)
            #record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim = 0)
            all_scores = torch.cat((all_scores, decoder_scores), dim = 0)
            # prepare the current tokento be the next decoder input (and add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)

        # finally, return the collections of word tokens and scores
        return all_tokens, all_scores

# evaluate function
# evaluate the response of chatbot based on the trained model that we have saved in directory
def evaluate(encoder, decoder, searcher, voc, sentence, max_length = handle_data.MAX_LENGTH):
    # change a batch of input sentence to index
    indexes_batch = [data_to_tensors.indexesFromSentence(voc, sentence)]
    # create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # transpose the dimensions of the input batch to match the model's dim expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0,1)
    # use approriate device
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    # decode sentence by using our searcher (GreedySearchDecoder, also written in this module)
    tokens, scores = searcher(input_batch, lengths, max_length)
    # change index to word
    decoded_words = [voc.indexToWord[token.item()] for token in tokens]
    return decoded_words

# evaluateInput function. calls evaluate() on our input query and generates its output sequence
# returns our output sentence in the conversation loop
def evaluateInput(encoder, decoder, searcher, voc):
    file_number = 1 # counts and makes files discernable by number for output TTS. initialize to 1
    print("\n -------------------- -={ CONVERSATION }=-  --------------------\n")
    while(1):
        try:
            # get the input query from our STT script to handle speech to text input
            input_sentence = str(STT.capture_input()) # this method from speechToText.py returns the captured voice input as text
            
            # test if input sentence is quit, if so, then quit. control flow goes back to our main script
            if input_sentence == 'q' or input_sentence == 'quit': break
            
            # normalize the string which takes a string and lowercases, trims, and removes non-letter chars
            input_sentence = normalizeString(input_sentence)
            print("Input ("+ str(file_number) + ") : "  + input_sentence)
            
            # generate the output through the evaluate function, also written in this module
            output_words = evaluate(encoder, decoder, searcher, voc, input_sentence) # list struct of words
            
            # adds each word in output_words into words if not 'eos' or 'pad'
            output_words[:] = [x for x in output_words if not (x=='EOS' or x== 'PAD')]
            print("Output (" + str(file_number) + ") : " + str(' '.join(output_words)))

            # implement our TTS script to handle text to speech output
            TTS.text_to_speech(str(' '.join(output_words)), file_number) # returns the output sequence to the input query
            file_number+=1 # increase file number to discern responses as mp3 files
                               
        except KeyError:  # in the case to handle unknown words
            print("Error: Encountered unknown word.")

    
 
