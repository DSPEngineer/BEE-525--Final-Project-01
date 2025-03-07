#! /usr/bin/python

import tensorflow as tf
import numpy as np

mnist = tf.keras.datasets.mnist

(train_data, train_labels), (test_data, test_labels) = mnist.load_data()
train_data, test_data = train_data / 255, test_data / 255

model = tf.keras.Sequential([
    tf.keras.layers.Flatten( input_shape=(28,28) ),
    tf.keras.layers.Dense( 128, activation=tf.nn.relu),
    tf.keras.layers.Dense( 10,  activation=tf.nn.softmax)
    ])


model.compile( optimizer = tf.keras.optimizers.Adam(),
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])

model.fit( train_data, train_labels, epochs=5 )

model.evaluate( test_data, test_labels)

predictions = model.predict(test_data)
np.set_printoptions(suppress=True)
print(test_labels[0])
print(predictions[0])
