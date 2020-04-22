# Abstractive-Text-Summarization-Project
First attempt at an NLP/NLU project. This was a project completed for our Artificial Intelligence course. The purpose of the project was to explore the space of NLP/NLU and get hands on experience working with deep learning as a tool. We used an Encoder-Decoder NN architecture to model emails and output their subjects. If we have time we would like to try summarizing different text as emails and their subjects sometimes don't appear related. We used Tensorflow and Keras for the deep learning.
### Define:
Text - Body of language to be summarized.  <br />
Summary - Distilled meaning of text.

## Model
The encoder model is a LSTM. The decoder is a GRU. No attention model in between. We hope to try to use attention given the time to do so.

## Dataset
https://www.tensorflow.org/datasets/catalog/aeslc

## Data Flow
### Preprocessing:
Data was tokenized by making it all lower case, removing punctuation and splitting on spaces. We did not remove stop words in an attempt to create natural English although this is something that may be tried in the future. We did not normalize the text as we were embedding the words using Glove (100d) and needed the words to be in the Glove file. The summaries were encoded into probability distributions which are essentially one-hot-encoding in the case of a training target. This is because there is a 1 in the word index and a 0 for the rest making that word 100% likely because it is a known training target. Both text and summaries were padded or truncated to a maximum number of words to keep the dimensions of the vectors consistent so they could be put into matrix form.
### Model Input (Encoder Input):
List of words in text as embedded word vectors.
### Encoder Output: 
Encoder model states.
### Decoder Input:
Encoder state + decoder output.
### Decoder Output:
Sequence of encoded word probabilities which can be translated back to words by getting the most probable (highest valued) index and mapping it using the `reverse_word_index` (which is the reverse of the `word_index`). We handled empty words, by getting the next most probably words.
### Model Prediction Process:
To predict the embedded tokens are inputted into the encoder. The encoder outputs its state to the decoder. The current decoder output is also passed to the decoder which is initialized as the start token. The decoder predicts one word per iteration with the encoder state and the current sequence. The decoder stops when it reaches the end token or the maximum desired length is reached. The model predictions are in the form of a probability distribution of all the words in our vocabulary as vector of probabilities. The most probable word is chosen, then one hot encoded with our current sequence which is then reinputted into the decoder. This process continues until an end word is reached, or the summary reaches the maximum desired length.

## Works Cited:
We learned from many sources. The code in particular was largely based on: https://github.com/chen0040/keras-text-summarization/blob/master/keras_text_summarization/library/seq2seq.py .  <br />
Another useful code source which we did not have time to implement, but train and test: https://www.kaggle.com/rahuldshetty/text-summarization-in-pytorch
