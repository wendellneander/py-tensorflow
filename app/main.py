import numpy as np
import tensorflow as tf
from tensorflow import keras


def example_1():
    model = keras.Sequential([
        keras.layers.Dense(units=1, input_shape=[1])
    ])

    model.compile(optimizer='sgd', loss='mean_squared_error')

    xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
    ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

    model.fit(xs, ys, epochs=500)

    predict = model.predict([10.0])
    print(predict)

    return predict


def example_2():
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=tf.nn.relu),
        keras.layers.Dense(128, activation=tf.nn.softmax),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer=tf.train.AdamOptimizer(), loss='sparse_categorical_crossentropy')

    model.fit(train_images, train_labels, epochs=5)

    test_loss, test_acc = model.evaluate(test_images, test_labels)

    my_images = []

    predictions = model.predict(my_images)


