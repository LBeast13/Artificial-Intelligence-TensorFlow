# -*- coding: utf-8 -*-
"""
Basic classification Tensorflow (AI)

Dataset used : Fashion Mnist

Each image :
    - is 28 x 28 px 
    - each pixel is represented by one value from 0 to 255
    - has one label

The labels are integers corresponding the class of clothing 
the image represent according :
    - 0 = T-shirt/top
    - 1 = Trouser
    - 2 = Pullover
    - 3 = Dress
    - 4 = Coat
    - 5 = Sandal
    - 6 = Shirt
    - 7 = Sneaker
    - 8 = Bag
    - 9 = Ankle boot
"""

from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# We store the class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def plot_image(i, predictions_array, true_label, img):
    """ 
    Display the image (img) with the label in :
        - blue = the prediction is correct
        - red = the prediction is wrong
    The label follow the pattern :
        Predicted Label confidence % (real label)
    """
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
  
    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
  
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100*np.max(predictions_array),
                                         class_names[true_label]),
                                         color=color)
    
def plot_value_array(i, predictions_array, true_label):
    """
    Display a bar graph of the predictions for the ith image coloring the
    prediction in red and the real label in blue.
    """
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1]) 
    predicted_label = np.argmax(predictions_array)
     
    thisplot[predicted_label].set_color('red')    # red for prediction
    thisplot[true_label].set_color('blue')        # blue for real label


def plot_several_images(predictions,test_labels, test_images,num_rows, num_cols):
    """
    Plot the first num_rows*num_cols test images, their predicted label, and the
    true label
    """
    print("---------")
    print("Predictions for the first ", num_rows*num_cols, " images")
    num_images = num_rows*num_cols
    plt.figure(figsize=(2*2*num_cols, 2*num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2*num_cols, 2*i+1)
        plot_image(i, predictions, test_labels, test_images)
        plt.subplot(num_rows, 2*num_cols, 2*i+2)
        plot_value_array(i, predictions, test_labels)
    plt.show()
    

def data_exploration(images, labels, class_names):
    """
    Displays information on the images and labels for a better understanding
    of the datasets.
    """
    images.shape      #Output the number of images and their size 
    len(labels)       #Output the number of labels
    labels            #Output the content of the labels
    
    # Plot the pixels of the first image in colorbar
    plt.figure()                    # Creates the new figure
    plt.imshow(images[0])     # Select the image to plot
    plt.colorbar()                  # Plot the image with a color bar
    plt.grid(False)                 # Hide the grid
    plt.show()                      # Display the image
    
    # Display the first 25 images from the training set with their class name
    plt.figure(figsize=(10,10)) 
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])              # Remove the x-axis values
        plt.yticks([])              # Remove the y-axis values
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[labels[i]])        # Display the class name
    plt.show()
    
    
def build_model(train_images, train_labels):
    """
    Builds, trains and return the model.
    1. Setup the Layers :
       - The first Layer (Flatten) reformats the data (2Darray-28x28) in a 
         1D-array-784 (28x28)
       - The second Layer (Dense) has 128 nodes (neurons)
       - The last Layer (Dense) returns an array of 10 probability scores that 
         sum to 1 Each node contains a score that indicates the probability that 
         the current image belongs to one of the 10 classes.
    2. Compile the model :
       - Loss function : measures how accurate the model is during training. 
         We want to minimize this function to "steer" the model in the right 
         direction.
       - Optimizer : how the model is updated based on the data it sees and 
         its loss function.
       - Metrics : to monitor the training and testing steps.     
    3. Train the model :
       - Feed the training data to the model—in (the train_images and 
         train_labels arrays)
       - The model learns to associate images and labels.
       - Ask the model to make predictions about a test set—in (test_images array) 
         We verify that the predictions match the labels from the test_labels array.
    """
    # BUILD 
    print("Start Building the model")
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),         # First Layer 
        keras.layers.Dense(128, activation=tf.nn.relu),     # Second Layer
        keras.layers.Dense(10, activation=tf.nn.softmax)    # Third Layer
    ])
    print("Model successfully built")
    print("--------------------")
    
    # COMPILE 
    print("Start compiling the model")
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])   # accuracy = the fraction of the images 
                                          # that are correctly classified.
    print("Model successfully compiled")
    print("--------------------")
    
    # TRAIN  
    print("Start training the model...")                                    
    model.fit(train_images, train_labels, epochs=5)
    print("Model successfully trained and ready to use")
    print("--------------------")
    
    return model

def make_predictions(model, test_images):
    """
    We get a prediction for each image.
    Each prediction consists in an array of 10 numbers representing the 
    "confidence" of the model that the image corresponds to each of the 10
    different articles of clothing.
    """
    
    predictions = model.predict(test_images)   # Make predictions
    
    print("Example of prediction for an image : ", predictions[0])                            
    print("The label with the highest confidence is number :", 
          np.argmax(predictions[0]))                 
    
    return predictions

def main():
    """
    The main function called by the entry point : 
    - Explore the data 
    - Preprocess the data
    - Build and train the model
    - Make predictions
    """
    # Loads the dataset returning 4 numPy arrays :
    #   - Train_images and train_labels --> the training set
    #   - test_images and test_labels --> the test set
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels),(test_images, test_labels) = fashion_mnist.load_data()
    
    
    # EXPLORING THE DATA (COMMENT THIS SECTION WHEN THE DATASET IS UNDERSTOOD)
    # data_exploration(train_images, train_labels, class_names)  # Training Set
    # data_exploration(test_images, test_labels, class_names)    # Test set
    
    # PREPROCESS THE DATA
    # We scale the pixel values between 0 to 1 of both datasets
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    
    #BUILD AND TRAIN MODEL
    model = build_model(train_images, train_labels)
    
    # Evaluate accurancy to compare accurancy on the training and test data
    print("---------------------")
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('Test accuracy:', test_acc)
    
    # MAKE AND DISPLAY PREDICTIONS
    predictions = make_predictions(model, test_images) 
    plot_several_images(predictions,test_labels,test_images,5,3)
  
if __name__== "__main__":
    """
    Entry point of the script.
    """
    main()