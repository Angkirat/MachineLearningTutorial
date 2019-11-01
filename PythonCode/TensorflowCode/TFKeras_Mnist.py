#!/Library/anaconda3/envs/MachineLearning/bin/python

"""
Copyright 2019 The TensorFlow Authors.
This is a modified piece of code copied from the Tensorflow learning link: https://www.tensorflow.org/overview/

This is a documented Hello world program for Tensorflow beginners.
It uses the MNIST data to show how to build a sequential categorical model using Tensorflow Keras.

v0.1 - Sequential Model with acc ~98%
"""
__author__ = "TensorFlow Authors, Angkirat Sandhu"
__source__ = \
    "https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/quickstart/beginner.ipynb"
__version__ = 0.1
__license__ = "Apache License, Version 2.0"
__email__ = "angkirat@gmail.com"
__status__ = "Prototype"
__maintainer__ = "Angkirat Sandhu"

# Importing the required packages
import tensorflow as tf

# Loading MNIST dataset from the Keras package. The data(both x and y) is split into train and test sets. `x`
# variables are normalized by dividing the whole set with 255(maximum integer value that can be held in grayscale
# images) to bring the data in the range of 0-1.
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
print("Training Set count {}".format(x_train.shape[0]))
print("Testing Set Count {}".format(x_test.shape[0]))

# Creating a Keras sequential model with a single hidden layer of 128 neurons and one dropout layer.
# The input layer takes in a np array of size (28,28).
# The output layer predicts probability of 10 classes.
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Finally the model is trained using the training set with 5 epoch(periods).
# The evaluate function directly shares the accuracy and loss of model over the test set
# without sharing the prediction result
model.fit(x_train, y_train, epochs=5)
loss, Acc = model.evaluate(x_test, y_test)
print("Loss value is {} Accuracy is {}".format(loss, Acc))
