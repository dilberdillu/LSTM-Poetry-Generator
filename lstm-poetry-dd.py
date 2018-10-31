# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 09:34:01 2018

@author: Dilber
"""

#importing all essential modules
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.optimizers import RMSprop
from keras.callbacks import LambdaCallback
from keras.callbacks import ReduceLROnPlateau

# takes the input file 
filename = "C:/Users/Dilber/Desktop/DL/poem_data3.txt"
raw_text = open(filename, encoding="utf8").read()
raw_text = raw_text.lower() #converts all character to lower case for simplicity

# takes all the unique characters from file and stores 
# in variable 'chars' as sorted list
chars = sorted(list(set(raw_text)))

# function that maps each character to an integer
# given a character/interger it returns correspending integer/character
char_to_int = dict((c,i) for i, c in enumerate(chars))
int_to_char = dict((i,c) for i, c in enumerate(chars))


n_chars = len(raw_text) # total #of characters in input file
n_vocab = len(chars)    # total unique characters in input file


max_len = 64   # length of a sentence that we use to train 
step = 3       # span of characters that we learn    
sentence = []  # to store sentences to train
next_char = [] # next character after the sentence


# at each iteration of this loop, we append a sentence of length 
# 'max_len' to the list 'sentence' and corresponding next character after 
# the sentence to the list 'next_char'
for i in range(0, n_chars - max_len, step):
    sentence.append(raw_text[i:i+max_len])
    next_char.append(raw_text[i+max_len])


# to represent our sequences as boolean values, we declare zero matices
# a 3dim matrix 'x' with #of sentences as row no. , length of a sentence as  
# column no. and vocabulary length as depth
x = np.zeros((len(sentence), max_len, len(chars)),dtype=np.bool)
y = np.zeros((len(sentence), len(chars)), dtype= np.bool)


# assigns value 1 to corresponding row/column to represent sentences as boolean matrices
for i, sentenc in enumerate(sentence):
    for t ,char in enumerate(sentenc):
        x[i, t, char_to_int[char]] = 1
    y[i, char_to_int[next_char[i]]] = 1 

# building a sequential model and adding layers to it
model = Sequential()
model.add(LSTM(128, input_shape = (max_len, len(chars)))) #LSTM layer with 128 units        
model.add(Dense(len(chars)))     #Final fully connected dense output layer 
model.add(Activation('softmax')) #activation function

optimizer = RMSprop(lr= 0.01)    # optimizer RMSprop with learning rate 0.01
model.compile(loss = 'categorical_crossentropy', optimizer = optimizer) #compiling

# helper function to sample an index from a probability array
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)



#saving our trained weights in disk by giving checkpoints
#reducing learning rate with a factor of 0.2 according to loss
filepath = "weights.hdfs"
print_callback = LambdaCallback()
checkpoint = ModelCheckpoint(filepath, monitor='loss',verbose=1, save_best_only=True, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2,patience=1, min_lr=0.001)
callbacks = [print_callback, checkpoint, reduce_lr]


# fitting our model
model.fit(x, y, batch_size=128, epochs=1, callbacks=callbacks)


# Text Generate Function
# this function takes an input text from user and reproduces a continuation
# text with given #of characters and temperature/diversity
def myGenerate(length_given, diversity_given):
    input_taken = []  #user input text is stored here
    sent = []           
    input_taken = input('Enter first line of poem (min 40 chars):  ')
    while(len(input_taken) < 64): # since the sentence length is predefined,
        input_taken = []          # a minimum character or 'max_val' is expected
        input_taken = input('..too short, please retype')
    sent = input_taken[0:max_len] # first characters upto value of 'max_len'
    gen = ''                      # is considered, to avoid input shape
    gen += sent                   # compatibility problem
    for i in range(length_given):
        x_predicted = np.zeros((1, max_len, len(chars))) 
        for t, ch in enumerate(sent):  # converts the user entered text to 
            x_predicted[0, t, char_to_int[ch]] = 1 # a matrix 'x_predicted'
        # and pass this matrix to model.predict() and stores return value in
        predictions = model.predict(x_predicted, verbose = 0)[0] # predictions
        # samples the character indices from helper function sample()
        next_ind = sample(predictions, diversity_given)
        next_ch = int_to_char[next_ind] # maps the index to characters
        gen += next_ch                  # appends the generated character
        sent = sent[1:] + next_ch       # appends to 'sent' to generate further
    return gen

print(myGenerate(500, 1.0)) # Generates text with given #of characters
                             # and text
