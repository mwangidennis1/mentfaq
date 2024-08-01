'''import h5py

# Open the HDF5 file
file_path = '/path/to/your/file.h5'  # Replace with your file path
with h5py.File(file_path, 'r') as hdf:
    # List all groups
    print("Keys: %s" % hdf.keys())
    # Get the first group name
    a_group_key = list(hdf.keys())[0]

    # Get the group object
    group = hdf[a_group_key]

    # List all datasets in the group
    print("Datasets: %s" % group.keys())

    # Read a dataset
    data = group['dataset_name'][:]  # Replace 'dataset_name' with the actual dataset name
    print(data)
'''
import nltk
import re
import random
import pandas as pd
import numpy as np
from tensorflow.keras.utils import plot_model
from keras.models import load_model
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.layers import Input, LSTM, Dense, Embedding, Attention
from keras.models import Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Constants
path_to_csv = 'C:\\Users\\Lenovo\\Downloads\\mentfaq\\Mental_Health_FAQ.csv'
dimensionality = 128  # Reduced dimensionality
batch_size = 32   # Increased batch size
epochs = 800  # Reduced epochs
embedding_dim = 100
max_vocab_size = 10000

# Load and preprocess data
data = pd.read_csv(path_to_csv, nrows=30)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

data['Questions'] = data['Questions'].apply(preprocess_text)
data['Answers'] = data['Answers'].apply(preprocess_text)

# Tokenization
special_tokens = ['<START>', '<END>']
tokenizer = Tokenizer(num_words=max_vocab_size, oov_token='<OOV>')
tokenizer.fit_on_texts(special_tokens + list(data['Questions'] + ' ' + data['Answers']))

# Print tokenizer word index to verify tokens
print(tokenizer.word_index)

# Convert text to sequences
encoder_input_data = tokenizer.texts_to_sequences(data['Questions'])
decoder_input_data = tokenizer.texts_to_sequences(['<START> ' + ans for ans in data['Answers']])
decoder_target_data = tokenizer.texts_to_sequences([ans + ' <END>' for ans in data['Answers']])

# Pad sequences
max_encoder_seq_length = max([len(seq) for seq in encoder_input_data])
max_decoder_seq_length = max([len(seq) for seq in decoder_input_data])

encoder_input_data = pad_sequences(encoder_input_data, maxlen=max_encoder_seq_length, padding='post')
decoder_input_data = pad_sequences(decoder_input_data, maxlen=max_decoder_seq_length, padding='post')
decoder_target_data = pad_sequences(decoder_target_data, maxlen=max_decoder_seq_length, padding='post')

# Convert target data to one-hot encoding
decoder_target_data = keras.utils.to_categorical(decoder_target_data, num_classes=max_vocab_size)

# Split the data
encoder_input_train, encoder_input_val, decoder_input_train, decoder_input_val, decoder_target_train, decoder_target_val = train_test_split(
    encoder_input_data, decoder_input_data, decoder_target_data, test_size=0.2, random_state=42)

# Encoder
encoder_inputs = Input(shape=(None,))
encoder_embedding = Embedding(max_vocab_size, embedding_dim, input_length=max_encoder_seq_length)(encoder_inputs)
encoder_lstm = LSTM(dimensionality, return_sequences=True, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(None,))
decoder_embedding = Embedding(max_vocab_size, embedding_dim, input_length=max_decoder_seq_length)(decoder_inputs)
decoder_lstm = LSTM(dimensionality, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)

# Attention layer
attention = Attention()([decoder_outputs, encoder_outputs])

# Concatenate attention output with decoder output
decoder_concat_input = keras.layers.Concatenate()([decoder_outputs, attention])

decoder_dense = Dense(max_vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_concat_input)

# Define the model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
print(model.summary())
'''
# Train the model
history = model.fit(
    [encoder_input_train, decoder_input_train], decoder_target_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=([encoder_input_val, decoder_input_val], decoder_target_val)
)

# Save the model
model.save('training_model.h5')
# Plot training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
'''
# Load the trained model
model = load_model('training_model.h5', custom_objects={'Attention': Attention})

# Print model summary to inspect the layer structure
model.summary()

# Create encoder model
encoder_inputs = model.input[0]
encoder_embedding = model.get_layer('embedding')
encoder_lstm = model.get_layer('lstm')
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding(encoder_inputs))
encoder_states = [state_h, state_c]
encoder_model = Model(encoder_inputs, [encoder_outputs] + encoder_states)

# Create decoder model
decoder_inputs = model.input[1]
decoder_embedding = model.get_layer('embedding_1')
decoder_lstm = model.get_layer('lstm_1')
attention_layer = model.get_layer('attention')
concat_layer = model.get_layer('concatenate')
decoder_dense = model.get_layer('dense')

decoder_state_input_h = Input(shape=(dimensionality,))
decoder_state_input_c = Input(shape=(dimensionality,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_hidden_state_input = Input(shape=(max_encoder_seq_length, dimensionality))

dec_emb = decoder_embedding(decoder_inputs)
decoder_outputs, state_h, state_c = decoder_lstm(dec_emb, initial_state=decoder_states_inputs)

attention_output = attention_layer([decoder_outputs, decoder_hidden_state_input])
decoder_concat_input = concat_layer([decoder_outputs, attention_output])

decoder_outputs = decoder_dense(decoder_concat_input)

decoder_model = Model(
    [decoder_inputs] + [decoder_hidden_state_input] + decoder_states_inputs,
    [decoder_outputs] + [state_h, state_c])

def decode_sequence(input_seq):
    # Encode the input as state vectors.
    encoder_outputs, state_h, state_c = encoder_model.predict(input_seq)
    
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0] = tokenizer.word_index.get('<START>', 0)  # Use default value if token is not found

    # Sampling loop for a batch of sequences
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + [encoder_outputs] + [state_h, state_c])

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = tokenizer.index_word.get(sampled_token_index, '')  # Use default value if index is not found
        decoded_sentence += ' ' + sampled_word

        # Exit condition: either hit max length or find stop character.
        if (sampled_word == '<END>' or len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        # Update states
        state_h, state_c = h, c

    return decoded_sentence

class ChatBot:
    negative_responses = ("no", "nope", "nah", "naw", "not a chance", "sorry")
    exit_commands = ("quit", "pause", "exit", "goodbye", "bye", "later", "stop")
    
    def start_chat(self):
        user_response = input("Hi, I'm a chatbot trained on mental health FAQs. How can I help you?\n")
        
        if user_response.lower() in self.negative_responses:
            print("Ok, have a great day!")
            return
        self.chat(user_response)
    
    def chat(self, reply):
        while not self.make_exit(reply):
            reply = input(self.generate_response(reply)+"\n")
    
    def generate_response(self, user_input):
        input_seq = tokenizer.texts_to_sequences([user_input])
        input_seq = pad_sequences(input_seq, maxlen=max_encoder_seq_length, padding='post')
        
        decoded_sentence = decode_sequence(input_seq)
        
        # Clean up the response
        decoded_sentence = decoded_sentence.replace('<START>', '').replace('<END>', '').strip()
        return decoded_sentence
    
    def make_exit(self, reply):
        for exit_command in self.exit_commands:
            if exit_command in reply.lower():
                print("Ok, have a great day!")
                return True
        return False

# Create and start the chatbot
chatbot = ChatBot()
chatbot.start_chat()
