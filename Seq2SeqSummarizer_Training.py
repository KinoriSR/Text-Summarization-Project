import tensorflow as tf
import json
import os
import numpy as np
# import pandas as pd
import sys
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import tensorflow_datasets.public_api as tfds
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding, Input
from keras.layers import LSTM, Flatten, Dense
from keras.layers import Conv1D, MaxPooling1D
from keras.models import model_from_json, load_model
import nltk


print('Using tensorlfow ', tf.__version__)

def create_dict(raw_data):
    data = []

    for example in raw_data: 
        email, subject = example["email_body"], example["subject_line"]

        d = {}
        a = email.numpy().decode("utf-8") 
        d['email'] = a.lstrip('b')
        h = subject.numpy().decode("utf-8") 
        d['subject'] = h.lstrip('b')
        data.append(d)
    return data

# open raw data
def load_data(data):
    raw = tfds.load(data)
    train_raw = raw['train']
    test_raw = raw['test']
    validate_raw = raw['validation']

    # list of dictionaries of data [{'email': email, 'subject': subject},{}]
    train = create_dict(train_raw)
    test = create_dict(test_raw)
    validate = create_dict(validate_raw)
    return train, test, validate

# split into text and summary : Emails
def split_email_text_summary(dataset):
    text = []
    summary = []
    for i in range(len(dataset)):
        text.append(dataset[i]['email'])
        summary.append('START ' + dataset[i]['subject'])# + ' END')
    return text, summary

# make text, summary into sequence
def make_sequences(tokenizer, text, summary, maxlen_text, maxlen_summary):
    # make sequences
    sequences_text = tokenizer.texts_to_sequences(text)
    sequences_summary = tokenizer.texts_to_sequences(summary)
    # pad sequences after words
    sum=0
    for i in range(len(sequences_summary)):
        sum+=len(sequences_summary[i])
    print('Average length: ',sum/len(sequences_summary))
    data_text = pad_sequences(sequences_text, maxlen=maxlen_text, padding='post')
    data_summary = pad_sequences(sequences_summary, maxlen=maxlen_summary, padding='post')
    
    return data_text, data_summary

# input sequenced summary, output one_hot_encoded (decoder input)
def one_hot_encode(seq_summary, word_index):
    length = len(seq_summary)
    maxlen_summary = len(seq_summary[0])
    word_index_length = len(word_index)
    decoder_in = np.zeros((length, maxlen_summary, word_index_length))
    decoder_out = np.zeros((length, maxlen_summary, word_index_length))
    for line in range(length):
        for word in range(maxlen_summary):
            #the big fox => [the,big,fox] => *[1,2,3]* => [[100],[010],[001]]
            decoder_in[line][word][seq_summary[line][word]] = 1.
            if word > 0:
                decoder_out[line][word-1][seq_summary[line][word]] = 1.
    return decoder_in, decoder_out

def load_glove(embedding_dim):
    glove_dir = 'glove.6B'

    embeddings_index = {}
    #f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'))
    f = open('glove.6B.100d.txt')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))

    # maps word_index to Glove vector 
    # embedding_matrix[index of token] = Glove vector => [vec]
    embedding_matrix = np.zeros((max_words, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if i < max_words:
            if embedding_vector is not None:
                # Words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
    return embedding_matrix, embeddings_index

# embedded input vectors
def embed_sequences(text, embedding_matrix, maxlen_text, maxlen_summary): # summary
    emb_text = np.zeros((len(text),maxlen_text,100))
    for i in range(len(text)):
        for j in range(maxlen_text):
            emb_text[i][j] = embedding_matrix[text[i][j]]
    return emb_text 

# configuration info
maxlen_text = 100  # max text length
maxlen_summary = 4 # max summary length
max_words = 5000  # max vocabulary size

print('Load data...')
train, test, validate = load_data('aeslc')

print('Create test, summary for train, test, validate...')
text_train, summary_train = split_email_text_summary(train)
text_test, summary_test = split_email_text_summary(test)
text_validate, summary_validate = split_email_text_summary(validate)
# *word* <=> index (sequence) <=>  embedded vector

print('Tokenize vocabulary...')
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(text_train + text_test + text_validate)
# *word* <=> index (sequence) <=>  embedded vector

print('Make word_index and reverse_word_index...')
word_index = {}
reverse_word_index = {}
for k, v in tokenizer.word_index.items():
    if v < max_words + 1:
        word_index[k] = v
        reverse_word_index[v] = k
print('Found %s unique tokens.' % len(word_index))

print('Sequencing...')
data_text_train, data_summary_train = make_sequences(tokenizer, text_train, summary_train, maxlen_text, maxlen_summary)
data_text_test, data_summary_test = make_sequences(tokenizer, text_test, summary_test, maxlen_text, maxlen_summary)
data_text_validate, data_summary_validate = make_sequences(tokenizer, text_validate, summary_validate, maxlen_text, maxlen_summary)
# one hot encode summaries Decoder Input/Ouput
data_summary_train_decoder_in, data_summary_train_decoder_out = one_hot_encode(data_summary_train , word_index)
data_summary_test_decoder_in, data_summary_test_decoder_out = one_hot_encode(data_summary_test, word_index)
data_summary_validate_decoder_in, data_summary_validate_decoder_out = one_hot_encode(data_summary_validate, word_index)
# word <=> *index (sequence)* <=>  embedded vector



print('Shape of data_text_train tensor:', data_text_train.shape)
print('Shape of data_summary_train tensor:', data_summary_train.shape)

print('Load Glove...')
embedding_dim = 100
embedding_matrix, embeddings_index = load_glove(embedding_dim)
# word <=> *index (sequence)* <=>  embedded vector

print('Embedding data...\n') # Encoder input
emb_data_text_train = embed_sequences(data_text_train, embedding_matrix, maxlen_text, maxlen_summary)
emb_data_text_test = embed_sequences(data_text_test, embedding_matrix, maxlen_text, maxlen_summary)
emb_data_text_validate = embed_sequences(data_text_validate, embedding_matrix, maxlen_text, maxlen_summary)
# emb_data_text_train, emb_data_summary_train = embed_sequences(data_text_train, data_summary_train, embedding_matrix, maxlen_text, maxlen_summary)
# emb_data_text_test, emb_data_summary_test = embed_sequences(data_text_test, data_summary_test, embedding_matrix, maxlen_text, maxlen_summary)
# emb_data_text_validate, emb_data_summary_validate = embed_sequences(data_text_validate, data_summary_validate, embedding_matrix, maxlen_text, maxlen_summary)
# word <=> index (sequence) <=>  *embedded vector*

'''
MODELING:
https://github.com/chen0040/keras-text-summarization/blob/master/keras_text_summarization/library/seq2seq.py
'''
GLOVE_EMBEDDING_SIZE = 100
HIDDEN_UNITS = 32 # MAYBE 64

print('Creating Model...\n')
encoder_inputs = Input(shape=(None, GLOVE_EMBEDDING_SIZE), name='encoder_inputs')
encoder_lstm = LSTM(units=HIDDEN_UNITS, return_state=True, name='encoder_lstm')
encoder_outputs, encoder_state_h, encoder_state_c = encoder_lstm(encoder_inputs)
encoder_states = [encoder_state_h, encoder_state_c]

decoder_inputs = Input(shape=(None, max_words), name='decoder_inputs')
decoder_lstm = LSTM(units=HIDDEN_UNITS, return_state=True, return_sequences=True, name='decoder_lstm')
decoder_outputs, decoder_state_h, decoder_state_c = decoder_lstm(decoder_inputs,
                                                                 initial_state=encoder_states)
decoder_dense = Dense(units=max_words, activation='softmax', name='decoder_dense')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

encoder_model = Model(encoder_inputs, encoder_states)


decoder_state_inputs = [Input(shape=(HIDDEN_UNITS,)), Input(shape=(HIDDEN_UNITS,))]
decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_state_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model([decoder_inputs] + decoder_state_inputs, [decoder_outputs] + decoder_states)
print(model.summary(), encoder_model.summary(),decoder_model.summary())

print('Fitting Model...\n')
epochs = 20

# print('decode_in shape: ', data_summary_test_decoder_in.shape)

model.fit([emb_data_text_train, data_summary_train_decoder_in], data_summary_train_decoder_out,
         batch_size = 32,
         epochs = epochs,
         verbose = 2,
         validation_data = ([emb_data_text_test, data_summary_test_decoder_in], data_summary_test_decoder_out))
         

print('Saving model in Encoder_Decoder_save1.h5...')
model.save("Encoder_Decoder_save1.h5")
print('saving encode model...')
encoder_model.save("Encoder_model_version1.h5")
print('saving Decoder model...')
decoder_model.save("Decoder_model_version1.h5")

trained_model = load_model('Encoder_Decoder_save1.h5')
trained_encoder_model = load_model('Encoder_model_version1.h5')
trained_decoder_model = load_model('Decoder_model_version1.h5')
print(trained_model.summary())
print(trained_encoder_model.summary())
print(trained_decoder_model.summary())

# Predict the first 10 in the test set
for summary in range(10):
    terminated = False
    target_text_len = 0
    target_text = ''
    target_seq = np.zeros((1, 1, max_words))
    target_seq[0, 0, word_index['start']] = 1
    input_seq = np.zeros(shape=(1, maxlen_text, GLOVE_EMBEDDING_SIZE))
    for i in range(maxlen_text):
        for j in range(GLOVE_EMBEDDING_SIZE):
            input_seq[0][i][j]=emb_data_text_validate[summary][i][j]
    states_value = trained_encoder_model.predict(input_seq)
    while not terminated:
        output_tokens, h, c = trained_decoder_model.predict([target_seq] + states_value)
        sample_token_idx = np.argmax(output_tokens[0, -1, :])

        #find second largest prob
        if sample_token_idx == 0:
            output_tokens[0, -1, sample_token_idx] = -1.
            #print(output_tokens)
            sample_token_idx = np.argmax(output_tokens[0, -1, :])
        sample_word = reverse_word_index[sample_token_idx]


        target_text_len += 1

        if sample_word != 'start' and sample_word != 'end':
            target_text += ' ' + sample_word

        if sample_word == 'end' or target_text_len >= maxlen_summary:
            terminated = True

        target_seq = np.zeros((1, 1, max_words))
        target_seq[0, 0, sample_token_idx] = 1

        states_value = [h, c]
    print('result: ', target_text)

