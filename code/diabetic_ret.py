import os
import zipfile
import matplotlib.pyplot as plt
import numpy as np
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import Conv2DLayer
from lasagne.layers import MaxPool2DLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import adam
from lasagne.layers import get_all_params
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import TrainSplit
from nolearn.lasagne import objective
import csv
import cv2
import scipy

print 1


def plot_image(image):
    plt.imshow(image)
    plt.show()


def load_data(folder_paths, labels_path, log=False, limit=-1):
    X = []
    y = []
    labels = {}
    with open(labels_path, 'rb') as csvfile:
        for row in csv.reader(csvfile):
            labels[row[0]] = row[1]
    count = 0
    for folder_path in folder_paths:
        for subdir, dirs, files in os.walk(folder_path):
            for file in files:
                count += 1
                path = os.path.join(subdir, file)
                if '.' not in file:
                    continue

                label, file_type = file.split('.')

                if log:
                    print "---File %s" % file
                image = cv2.imread(path)

                if log:
                    print "\t--Resizing..."
                image_resized = cv2.resize(image, (300, 300))

                try:
                    y.append(labels[label])
                    X.append(image_resized)
                except KeyError:
                    continue

                # TODO: Normalization
                if log:
                    print "\n"

                if count == limit:
                    break

    if log:
        print "-Converting"

    # Theano works with fp32 precision
    X = np.array(X).astype(np.float32)
    y = np.array(y).astype(np.int32)

    if log:
        print "-Reshaping"

    X = X.reshape(
        -1,  # number of samples, -1 makes it so that this number is determined automatically
        3,  # 1 color channel, since images are only black and white
        300,  # first image dimension (vertical)
        300,  # second image dimension (horizontal)
    )

    return X, y


folder_paths = [os.path.relpath('../data/diabetic_ret/train_001/')]
labels_path = os.path.relpath('../data/diabetic_ret/trainLabels.csv')

X, y = load_data(folder_paths, labels_path, limit=100, log=True)

print "DONE"

layers0 = [
    # layer dealing with the input data
    (InputLayer, {'shape': (None, X.shape[1], X.shape[2], X.shape[3])}),

    # first stage of our convolutional layers
    (Conv2DLayer, {'num_filters': 96, 'filter_size': 5}),
    (Conv2DLayer, {'num_filters': 96, 'filter_size': 3}),
    (Conv2DLayer, {'num_filters': 96, 'filter_size': 3}),
    (Conv2DLayer, {'num_filters': 96, 'filter_size': 3}),
    (Conv2DLayer, {'num_filters': 96, 'filter_size': 3}),
    (MaxPool2DLayer, {'pool_size': 2}),

    # second stage of our convolutional layers
    (Conv2DLayer, {'num_filters': 128, 'filter_size': 3}),
    (Conv2DLayer, {'num_filters': 128, 'filter_size': 3}),
    (Conv2DLayer, {'num_filters': 128, 'filter_size': 3}),
    (MaxPool2DLayer, {'pool_size': 2}),

    # two dense layers with dropout
    (DenseLayer, {'num_units': 64}),
    (DropoutLayer, {}),
    (DenseLayer, {'num_units': 64}),

    # the output layer
    (DenseLayer, {'num_units': 10, 'nonlinearity': softmax}),
]


def regularization_objective(layers, lambda1=0., lambda2=0., *args, **kwargs):
    # default loss
    losses = objective(layers, *args, **kwargs)
    # get the layers' weights, but only those that should be regularized
    # (i.e. not the biases)
    weights = get_all_params(layers[-1], regularizable=True)
    # sum of absolute weights for L1
    sum_abs_weights = sum([abs(w).sum() for w in weights])
    # sum of squared weights for L2
    sum_squared_weights = sum([(w ** 2).sum() for w in weights])
    # add weights to regular loss
    losses += lambda1 * sum_abs_weights + lambda2 * sum_squared_weights
    return losses


net0 = NeuralNet(
    layers=layers0,
    max_epochs=10,

    update=adam,
    update_learning_rate=0.0002,

    objective=regularization_objective,
    objective_lambda2=0.0025,

    train_split=TrainSplit(eval_size=0.25),
    verbose=1,
)

net0.fit(X, y)
