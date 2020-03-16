		  -={ TO RUN THE CODE }=-
Run the file "main.py" inside ./src/ by using "python3 main.py". It will enter a conversation loop based on the trained models inside ./dataGOOD/




		-={ SCRIPTS INSIDE ./src/ }=-
./src/data_to_tensors.py - converts our list of sentence pairs into tensors to be fed into our RNNs

./src/handle_data.py - takes our .json training dataset and constructs a list of sentence pairs as well as our vocabulary object

./src/main.py - our main script. RUN THIS 

./src/rnn_models.py - contains our encoder rnn, decoder rnn, and luong global attention layer

./src/speech_to_text.py - handles the conversion of our speech to text inputs

./src/text_to_speech.py - handles the conversion of our text to speech outputs

./src/train_and_evaluation.py - contains our training and evaluation functions to be used with our models

./src/vocab.py - contains our vocabulary object's class / used to trim rare words and for word embeddings
