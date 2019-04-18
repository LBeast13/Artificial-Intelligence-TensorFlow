# -*- coding: utf-8 -*-
"""
Text Classification TensorFlow (AI)

Dataset used : Internet Movie Database IMDB

Classifies movie reviews as positive or negative (Binary classification with 
supervised learning) using the text of the review.

Each movie review :
    . Array of integers (each mapping a word in a dictionary)
    . have not a standard length

Each label is an integer :
    . 0 = Negative
    . 1 = Positive

@author: Liam Bette
"""

from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt


def get_dictionary(imdb):
    """
    Returns two dictionaries useful for coding/decoding :
        . One mapping a word to an index
        . Another mapping an index to a word
    """
    # A dictionary mapping words to an integer index
    word_index = imdb.get_word_index()
    
    # The first indices are reserved
    word_index = {k:(v+3) for k,v in word_index.items()} 
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2  # unknown
    word_index["<UNUSED>"] = 3
    
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    
    return word_index, reverse_word_index


def decode_review(text, decoder):
    """
    Return the decoded review using the dictionnary mapping integers to words
    """
    return ' '.join([decoder.get(i, '?') for i in text])


def data_exploration(data, labels):
    """
    Displays information on the images and labels for a better understanding
    of the datasets. We can learn that :
        . The data is composed by 25000 reviews and the there are also 25000 
          labels
        . Each review is represented by a list of integers where each integer 
          represents a specific word in a dictionary.
        . Reviews can have different lengths
    """
    print("Training entries: {}, labels: {}".format(len(data), len(labels)))
    print(labels)
    print(data[0])                      # The first review
    print(len(data[0]), len(data[1]))   # Compare the length of different reviews

def prepare_data(data,word_index):
    """
    Prepare the data by making all the arrays (reviews) the same length (256).
    """
    data = keras.preprocessing.sequence.pad_sequences(data,
                                                      value=word_index["<PAD>"],
                                                      padding='post',
                                                      maxlen=256)
    return data

def build_model(data, labels):
    """
    Build and returns the model and the history of the model.
    Layers :
        1. Embedding layer and it takes the integer-encoded vocabulary and 
           looks up the embedding vector for each word-index
        2. GlobalAveragePooling1D layer and returns a fixed-length output vector 
           for each example by averaging over the sequence dimension (allows 
           the model to handle input of variable length)
        3. Fully-connected (Dense) layer with 16 hidden units 
        4. Fully-connected (Dense) layer connected to a single output node 
           (the sigmod activation function is a value between 0 and 1 representing
           a level of confidence)
    Compilation :
        We use the binary_crossentropy loss function (better when dealing with
        probabilities)
    """
    
    # input shape is the vocabulary count used for the movie reviews (10,000 words)
    vocab_size = 10000
    
    # Layers Setup
    print("Start Building the model")
    model = keras.Sequential()
    model.add(keras.layers.Embedding(vocab_size, 16))           # First layer
    model.add(keras.layers.GlobalAveragePooling1D())            # Second layer
    model.add(keras.layers.Dense(16, activation=tf.nn.relu))    # Third layer
    model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))  # Fourth layer
    print("Model successfully built")
    print("--------------------")
    #model.summary() # Display a summary of the layers of the model
   
    # Model compilation
    print("Start compiling the model")
    model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc'])
    print("Model successfully compiled")
    print("--------------------")
    
    # Validation set to check the accuracy during training
    x_val = data[:10000]                # validation data
    partial_x_train = data[10000:]      # partial data for training
    
    y_val = labels[:10000]              # validation labels
    partial_y_train = labels[10000:]    # partial labels for training
    
    # Training the model
    print("Start training the model...")    
    history = model.fit(partial_x_train,
                        partial_y_train,
                        epochs=40,        # Number of times the data will pass in the network
                        batch_size=512,   # Each time we pass data we pass 512 reviews
                        validation_data=(x_val, y_val),
                        verbose=1)
    print("Model successfully trained and ready to use")
    print("--------------------")
    
    history_dict = history.history
    # print(history_dict.keys())    # The keys are : 'loss', 'acc', 'val_loss', 
                                    #                'val_acc
    
    return model, history_dict

def accuracy_loss_graph(history_dict):
    acc = history_dict['acc']           # Accurancy 
    val_acc = history_dict['val_acc']   # Accurancy (validation set)
    loss = history_dict['loss']         # Loss
    val_loss = history_dict['val_loss'] # Loss (validation set)
    
    epochs = range(1, len(acc) + 1)     # Number of epochs
    
    plt.plot(epochs, loss, 'bo', label='Training loss')  # "bo" = "blue dot"
    plt.plot(epochs, val_loss, 'b', label='Validation loss') # b = "solid blue line"
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.show()
    
    plt.clf()   # clear figure

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.show()

def main():
    """
    The main function called by the entry point : 
    """
    # DOWNLOAD THE DATA
    imdb = keras.datasets.imdb      # Load the movies dataset From imdb
    (train_data, train_labels),(test_data, test_labels) = imdb.load_data(num_words=10000)

    # EXPLORING THE DATA
    #data_exploration(train_data, train_labels)
    
    # CODE/DECODE DICTIONARIES
    word_index, reverse_word_index = get_dictionary(imdb)
    #print(decode_review(train_data[0], reverse_word_index))    # Test for decode
    
    # PREPARE THE DATA
    train_data = prepare_data(train_data, word_index)
    test_data = prepare_data(test_data, word_index)
    print("Data prepared with standard length for reviews.")
    print("--------------------")
    # print(train_data[0])
    
    # BUILD THE MODEL
    model, history_dict = build_model(train_data, train_labels)

    # Evaluate the built model    
    results = model.evaluate(test_data, test_labels)
    print(results) # Accuracy
    
    # Display an accuracy graph
    accuracy_loss_graph(history_dict)
    

if __name__== "__main__":
    """
    Entry point of the script.
    """
    main()
