# Artificial Intelligence with TensorFlow
>This repository is composed by different Deep Learning models built with [TensorFlow](https://www.tensorflow.org/) 
 using [tf.keras](https://www.tensorflow.org/api_docs/python/tf/keras) and based on the tutorials that you can find just
[here](https://www.tensorflow.org/tutorials)

## Basic Classification (Clothing Images)
![](basic_classification.png)


Corresponding script :
```
clothing_images_classification.py
```
Used dataset : [Fashion Mnist](https://github.com/zalandoresearch/fashion-mnist)

In this script, we train a neural network model (supervised learning) to classify images of clothing, like sneakers, shirts, coats...
Each image is 28x28 px where each pixel is represented by a value between 0 and 255.
Furthermore, each image has a label corresponding to the type of clothing (sneakers, sandals...)
There are 10 different labels :
* 0 = T-shirt/top
* 1 = Trouser
* 2 = Pullover
* 3 = Dress
* 4 = Coat
* 5 = Sandal
* 6 = Shirt
* 7 = Sneaker
* 8 = Bag
* 9 = Ankle boot

## Text Classification (Movie reviews)
![](text_classification.jpg)  
  
Corresponding script :
```
movies_reviews_classification.py
```
Used dataset : [Internet Movie Database IMDB](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/imdb)  

In this script, we train a neural network model (Binary classification with supervised learning) to classify reviews in two categories : positive and negative.
Each review is composed by several integers each representing a specific word in a dictionary. We are provided a map with a word mapped to an integer that we can easily reverse in order to code/decode any review.
All the reviews are acompanied by a label which can be :
* 0 = "Negative"
* 1 = "Positive"

## Regression (Cars fuel efficiency)
![](regression.png)   

Corresponding script :
```
fuel_efficiency_regression.py
```
Used dataset : [Auto MPG](https://archive.ics.uci.edu/ml/datasets/auto+mpg)

This script builds and trains a neural network (Supervised learning) to predict the fuel efficiency (MPG) of late-1970s and early 1980s automobiles (Regression problem).
We used the early stop in the training in order to terminate it when the model was not improving anymore.
