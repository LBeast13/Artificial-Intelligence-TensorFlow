# -*- coding: utf-8 -*-
"""
Text Classification TensorFlow (AI) : Movie reviews (Overfitting)

Dataset used : Internet Movie Database IMDB

Classifies movie reviews as positive or negative (Binary classification with 
supervised learning) using the text of the review.
This script shows the overfitting when training a model and to methods to
prevent it :
    - Reduce the capacity of the network
    
    - Weight regularization = we add to the loss function of the network a cost 
    associated with having large weights). We used here weight decay (L2 
    regularization) where the cost added is proportional to the square of the 
    value of the weights coefficients.
    
    - Dropout = consists of randomly "dropping out" (i.e. set to zero) a number 
    of output features of the layer during training. The "dropout rate" is the 
    fraction of the features that are being zeroed-out; it is usually set 
    between 0.2 and 0.5.At test time, no units are dropped out, and instead the 
    layer's output values are scaled down by a factor equal to the dropout rate

@author: Liam Bette
"""

from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

NUM_WORDS = 10000


def multi_hot_data(data, dimension):
    """
    Multi-hot-encoding the data (list of integers) means turning them into 
    vectors of 0s and 1s. 
    Example :
        The sequence [3, 5] would be converted into a 10,000-dimensional vector 
        that would be all-zeros except for indices 3 and 5, which would be ones.
    """
    # Create an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(data), dimension))
    
    for i, word_indices in enumerate(data): 
        results[i, word_indices] = 1.0  # set specific indices of results[i] to 1s
    return results

def build_model(layer_size, train_data, train_labels, test_data, test_labels):
    """
    Build and trains the model setting 2 Dense layers with a size defined by 
    the parameter layer_size.
    Returns the trained model and the history of the training
    """
    
    # Build the model
    model = keras.Sequential([ 
        keras.layers.Dense(layer_size, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
        keras.layers.Dense(layer_size, activation=tf.nn.relu),
        keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])
    
    # Compile the model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy', 'binary_crossentropy'])
    
    # Train the model
    history = model.fit(train_data,
                        train_labels,
                        epochs=20,
                        batch_size=512,
                        validation_data=(test_data, test_labels),
                        verbose=2)
    
    # Display the summary of the layers composing the model
    model.summary()
    
    return model, history


def build_model_L2(layer_size, train_data, train_labels, test_data, test_labels):
    """
    Build and trains the model setting 2 Dense layers with a size defined by 
    the parameter layer_size.
    We use a regularizer for the weight regularization.
    Returns the trained model and the history of the training
    """
    # Build the L2 regularized model
    model = keras.models.Sequential([
                keras.layers.Dense(layer_size, 
                                   kernel_regularizer=keras.regularizers.l2(0.001),
                                   activation=tf.nn.relu, 
                                   input_shape=(NUM_WORDS,)),
                keras.layers.Dense(layer_size, 
                                   kernel_regularizer=keras.regularizers.l2(0.001),
                                   activation=tf.nn.relu),
                keras.layers.Dense(1, activation=tf.nn.sigmoid)
                ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy', 'binary_crossentropy'])

    # Train the model
    history = model.fit(train_data, train_labels,
                              epochs=20,
                              batch_size=512,
                              validation_data=(test_data, test_labels),
                              verbose=2)
    return model, history

def build_model_dropout(layer_size, train_data, train_labels, test_data, test_labels):
    """
    Build and trains the model setting 2 Dense layers with a size defined by 
    the parameter layer_size.
    Use the dropout regularization technique.
    Returns the trained model and the history of the training
    """
    # Build the model with dropouts
    model = keras.models.Sequential([
            keras.layers.Dense(layer_size, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
            keras.layers.Dropout(0.5),      # Add a dropout on the first layer
            keras.layers.Dense(layer_size, activation=tf.nn.relu),
            keras.layers.Dropout(0.5),      # Add a dropout on the second layer
            keras.layers.Dense(1, activation=tf.nn.sigmoid)
            ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy','binary_crossentropy'])

    # Train the model
    history = model.fit(train_data, train_labels,
                        epochs=20,
                        batch_size=512,
                        validation_data=(test_data, test_labels),
                        verbose=2)
    
    return model, history
    

def plot_history(histories, key='binary_crossentropy'):
    """
    Plot the loss of the different models (training and validation)
    NB : a lower validation loss indicates a better model
    """
    plt.figure(figsize=(16,10))
    
    for name, history in histories:
        val = plt.plot(history.epoch, history.history['val_'+key],
                       '--', label=name.title()+' Val')
        plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
                 label=name.title()+' Train')

    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_',' ').title())
    plt.legend()

    plt.xlim([0,max(history.epoch)])


def main():
    """
    The main function called by the entry point:
        - Loads the data
        - Multi-hot encode the data
        - Plot the data to understand the encoding
        - Build different models :
            - A baseline model as reference
            - A smaller model
            - A bigger model
            - An L2 regularized model
            - A model with dropouts
        - Plot histograms of the training of the different models to compare them
    """
    
    # LOAD THE DATA
    (train_data, train_labels), (test_data, test_labels) = keras.datasets.imdb.load_data(num_words=NUM_WORDS)
    
    # MULTI HOT ENCODING THE DATA (0s and 1s)
    train_data = multi_hot_data(train_data, dimension=NUM_WORDS)
    test_data = multi_hot_data(test_data, dimension=NUM_WORDS)
    
    # Plot the data to understand the transformation
    # NB : wordindexes are sorted by frequency so it is normal that we have more
    # 1s near the index 0
    plt.plot(train_data[0])
    
    # BUILD MODELS (baseline, smaller, bigger, l2)
    baseline_model, baseline_history = build_model(16, 
                                                   train_data, train_labels, 
                                                   test_data, test_labels)
    smaller_model, smaller_history = build_model(4, 
                                                 train_data, train_labels, 
                                                 test_data, test_labels)
    
    bigger_model, bigger_history = build_model(512, 
                                               train_data, train_labels, 
                                               test_data, test_labels)
    
    # This model uses weight regularization to reduce overfitting
    l2_model, l2_history = build_model_L2(16, 
                                          train_data, train_labels, 
                                          test_data, test_labels)
    
    # This model uses dropout regularization technique to reduce overfitting
    dpt_model, dpt_history = build_model_dropout(16, 
                                                 train_data, train_labels, 
                                                 test_data, test_labels)
    
            
    # PLOT THE HISTORIES OF THE MODEL
    
    # As we can see, larger network begins overfitting almost right away, after 
    # just one epoch, and overfits much more severely. The more capacity the 
    # network has, the quicker it will be able to model the training data 
    # (resulting in a low training loss), but the more susceptible it is to 
    # overfitting (resulting in a large difference between the training and 
    # validation loss).
    plot_history([('baseline', baseline_history),
                  ('smaller', smaller_history),
                  ('bigger', bigger_history)])
    
    # We can see that L2 regularized model has become much more resistant to 
    # overfitting than the baseline model, even though both models have the same 
    # number of parameters.
    plot_history([('baseline', baseline_history),
                  ('l2', l2_history),])
    
    # We can see that model regularized with dropouts has also become much more 
    # resistant to overfitting than the baseline model
    plot_history([('baseline', baseline_history),
                  ('dropout', dpt_history)])
    
    
if __name__== "__main__":
    """
    Entry point of the script.
    """
    main()
