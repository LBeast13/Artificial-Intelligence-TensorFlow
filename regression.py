# -*- coding: utf-8 -*-
"""
Regression TensorFlow (AI)

Dataset used : Auto MPG

Train a model (Supervised Learning) to predict the fuel efficiency (MPG) of 
late-1970s and early 1980s automobiles.
We worked on data preparation and normalization for a Regression problem.
We used an early stop to finish the training of the model when it was not 
improving anymore.

Good to know : MPG = Miles Per Gallon

@author: Liam Bette
"""

from __future__ import absolute_import, division, print_function

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

pd.options.mode.chained_assignment = None   # Removes unwanted warnings

# The column names of the dataset
column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                   'Acceleration', 'Model Year', 'Origin']

 
class PrintDot(keras.callbacks.Callback):
    """
    Display training progress by printing a single dot for each completed epoch
    """
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')

def load_dataset():
    """
    Downloads the dataset and imports it using pandas. 
    """
    dataset_path = keras.utils.get_file("auto-mpg.data", "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
   
    # Imports the dataset using pandas
    raw_dataset = pd.read_csv(dataset_path, names=column_names,
                              na_values = "?", comment='\t',
                              sep=" ", skipinitialspace=True)
    dataset = raw_dataset.copy()
    
    return dataset

def explore_data(data):
    """
    Explores and displays information about the data to understand it better.
    """
    print(data.tail())          # Display the last 5 elements of the dataset
    
    print(data.isna().sum())    # Display the number of unknown elements for 
                                # each column
                                

def clean_data(data):
    """
    Cleans the data :
        - Drops all the unknown values
        - Converts Origin column in a one-hot (because it is a categorical column)
    """
    data = data.dropna()    # Deletes the unknown values
    
    origin = data.pop('Origin')  # Remove the column Origin and put it into origin

    data['USA'] = (origin == 1)*1.0     # New column USA for origin = 1
    data['Europe'] = (origin == 2)*1.0  # New column Europe for origin = 2
    data['Japan'] = (origin == 3)*1.0   # New column Japan for origin = 3
    print(data.tail())      # Display the modified data tail
    
    return data

def split_data(data):
    """
    Splits the data in two sets :
        - Training set (80% of the data)
        - Test set (20% of the data)
    """
    train = data.sample(frac=0.8,random_state=0)    # 80% Sample of the data
    test = data.drop(train.index)                   # the other 20% of the data

    return train, test

def visualize_data(data):
    """
    Displays graphs for the specified columns :
        - MPG
        - Cylinders
        - Displacement
        - Weight
    Also Displays stats on the data
    """
    sns.pairplot(data[["MPG", 
                       "Cylinders", 
                       "Displacement", 
                       "Weight"]], diag_kind="kde")     # Displays the graphs
    
    train_stats = data.describe()           # Gets stats on the data
    train_stats.pop("MPG")                  # Removes the MPG column
    train_stats = train_stats.transpose()   # Transpose the data
    print(train_stats) 

    return train_stats             

def norm(data,train_stats):
    """
    Normalizes the data using the training statistics
    """
    return (data - train_stats['mean']) / train_stats['std']

def build_model(train_dataset, normed_train_data, train_labels):
    """
    Builds a sequential model with:
        - 2 densely connected (Dense) layers
        - 1 Output layer returning a single continuous value
    Then trains the model using an early stop in order to stop the training
    when the model is not improving anymore.
    Returns the trained model and the history of the training
    """
    # Layers setup
    model = keras.Sequential([
        layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)  # Optimizer definition

    # Model Compilation
    model.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=['mean_absolute_error', 'mean_squared_error'])
    
    model.summary()     # Displays a summary of the layers of the model
    
    EPOCHS = 1000
    
    # This is to stop the training when the model does not improve anymore
    # The patience parameter is the amount of epochs to check for improvement
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    # Model Training
    history = model.fit(normed_train_data, train_labels,
                        epochs = EPOCHS, validation_split = 0.2, verbose=0,
                        callbacks=[early_stop,PrintDot()])
    
    return model, history

def plot_history(history):
    """
    Plots the history of the training.
    """
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
  
    # FIRST FIGURE (Mean Abs Error MPG)
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mean_absolute_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
             label = 'Val Error')
    plt.ylim([0,5])
    plt.legend()
  
    # SECOND FIGURE (Mean Square Error MPG^2)
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mean_squared_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'],
             label = 'Val Error')
    plt.ylim([0,20])
    plt.legend()
    plt.show()

def make_prediction(model, normed_test_data, test_labels):
    """
    Makes predictions on the normed_test_data using the model.
    Displays two figures using the predictions:
        - A scatter plot f(labels) = predictions
        - An histogram of the error distribution
    """
    # flatten() returns a copy of the array collapsed into one dimension
    test_predictions = model.predict(normed_test_data).flatten() 

    # Figure f(true values) = predictions
    # Should be near to the identity line (y=x)
    plt.scatter(test_labels, test_predictions)
    plt.xlabel('True Values [MPG]')
    plt.ylabel('Predictions [MPG]')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([0,plt.xlim()[1]])
    plt.ylim([0,plt.ylim()[1]])
    plt.plot([-100, 100], [-100, 100])  # Plot an identity line (y=x)

    plt.show()
    
    plt.clf()   # clear figure

    # Figure of the error distribution
    error = test_predictions - test_labels
    plt.hist(error, bins = 25)
    plt.xlabel("Prediction Error [MPG]")
    plt.ylabel("Count")

def main():
   """
   The main function called by the entry point:
       - Loads the data
       - Explores the data to understand it better
       - Cleans the data
       - Splits the data in train and test set and in train labels and test labels
       - Normalizes the data
       - Builds and trains the model
       - Makes predictions with model and displays the results
   """
   
   # LOAD THE DATASET
   dataset = load_dataset()
    
   # EXPLORE THE DATA
   #explore_data(dataset)
   
   # CLEAN THE DATA
   data = clean_data(dataset)
   
   # SPLIT THE DATA IN TRAIN AND TEST SET
   train_dataset, test_dataset = split_data(data)
   
   # VISUALIZE THE DATA
   train_stats = visualize_data(train_dataset)
   
   # SPLIT LABELS (MPG) FROM FEATURES
   train_labels = train_dataset.pop('MPG')  # Train labels
   test_labels = test_dataset.pop('MPG')    # Test labels
   
   # NORMALIZE THE DATA 
   # (Not necessary but makes the training easier and removes the dependency of 
   # the used units)
   normed_train_data = norm(train_dataset, train_stats)     # Normalize train set
   normed_test_data = norm(test_dataset, train_stats)       # Normalize test set
    
   # BUILD THE MODEL
   model, history = build_model(train_dataset, normed_train_data, train_labels)
   
   # Training stats
   plot_history(history)
   
   # Check the expected error that we could get in real situation using the test set
   loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=0)
   print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))
   
   # MAKE PREDICTIONS
   make_prediction(model, normed_test_data, test_labels)
   
if __name__== "__main__":
    """
    Entry point of the script.
    """
    main()
