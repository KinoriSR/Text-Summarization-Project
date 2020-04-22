import tensorflow as tf
import json
import os
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.models import model_from_json, load_model

def preprocess(text, maxlen_text):
    remove_list=' ! " # $ % & ( ) * + , - . / : ; < = > ? @ [ \\ ] ^ _ ` { | } ~ \t \n'.split(' ')
    text = text.lower()
    for i in range(len(remove_list)):
        text = text.replace(remove_list[i], '')
    data_text = text.split(' ')
    if len(data_text) > maxlen_text:
        data_text = data_text[0:maxlen_text]
    else:
        # pad sequences after words
        empty = []
        for j in range(maxlen_text - len(data_text)):
            empty.append('')
        data_text = data_text + empty
    return data_text

def load_glove(embedding_dim):
    glove_dir = 'glove.6B'

    embeddings_index = {}
    f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    # print('Found %s word vectors.' % len(embeddings_index))

    # maps word_index to Glove vector 
    # embedding_matrix[index of token] = Glove vector
    embedding_matrix = np.zeros((max_words, embedding_dim))
    for n, word in reverse_word_index.items():
        i = int(n)
        embedding_vector = embeddings_index.get(word)
        if i < max_words:
            if embedding_vector is not None:
                # Words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
    return embedding_matrix, embeddings_index

# embedded input vectors
def embed_sequences(text, embedding_index, maxlen_text, embedding_dim):
    emb_text = np.zeros((len(text),maxlen_text, embedding_dim))
    for i in range(maxlen_text):
        if text[i] in embedding_index:
            emb_text[i] = embedding_index[text[i]]
        else:
            emb_text[i] = np.zeros(embedding_dim)
    return emb_text

# configuration info
maxlen_text = 100  # text length
maxlen_summary = 4 # summary length
max_words = 5000  # vocabulary size
embedding_dim = 100

# print('Load data...')

# ***FROM FRONT END USER INPUT***
validate = input('Give me an input:')

# print('Preprocess...')
data_text_validate = preprocess(validate, maxlen_text)
# print(data_text_validate)

# load word_index 
File = open('reverse_word_index.json', 'r')
reverse_word_index = json.load(File)
# print('Make word_index and reverse_word_index...')
# print('Found %s unique tokens.' % len(reverse_word_index))

# print('Load Glove...')
embedding_matrix, embedding_index = load_glove(embedding_dim)
# word <=> *index (sequence)* <=>  embedded vector

# print('Embedding data...\n') 
# Encoder input
emb_data_text_validate = embed_sequences(data_text_validate, embedding_index, maxlen_text, embedding_dim)
# word <=> index (sequence) <=>  *embedded vector*

GLOVE_EMBEDDING_SIZE = 100
HIDDEN_UNITS = 32 # MAYBE 64

trained_model = load_model('Encoder_Decoder_save1.h5')
trained_encoder_model = load_model('Encoder_model_version1.h5')
trained_decoder_model = load_model('Decoder_model_version1.h5')
# print(trained_model.summary())
# print(trained_encoder_model.summary())
# print(trained_decoder_model.summary())

for k, v in reverse_word_index.items():
    if v == 'start':
        start = int(k)

terminated = False
target_text_len = 0
target_text = ''
target_seq = np.zeros((1, 1, max_words))
target_seq[0, 0, start] = 1
input_seq = np.zeros(shape=(1, maxlen_text, GLOVE_EMBEDDING_SIZE))
for i in range(maxlen_text):
    for j in range(GLOVE_EMBEDDING_SIZE):
        input_seq[0][i][j]=emb_data_text_validate[0][i][j]
states_value = trained_encoder_model.predict(input_seq)
while not terminated:
    output_tokens, h, c = trained_decoder_model.predict([target_seq] + states_value)
    sample_token_idx = np.argmax(output_tokens[0, -1, :])

    #find second largest prob
    if sample_token_idx == 0:
        output_tokens[0, -1, sample_token_idx] = -1.
        #print(output_tokens)
        sample_token_idx = np.argmax(output_tokens[0, -1, :])

        #sample_token_idx = np.argmax(output_tokens[0, -1, :])
    sample_word = reverse_word_index[str(sample_token_idx)]


    target_text_len += 1

    if sample_word != 'start' and sample_word != 'end':
        target_text += ' ' + sample_word

    if sample_word == 'end' or target_text_len >= maxlen_summary:
        terminated = True

    target_seq = np.zeros((1, 1, max_words))
    target_seq[0, 0, sample_token_idx] = 1

    states_value = [h, c]

# ***OUTPUT TO FRONT END***
print('result: ', target_text)
