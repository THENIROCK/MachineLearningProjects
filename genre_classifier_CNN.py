
# A note on altering code: The guidelines for this project mention 
# altering an existing codebase. However, for this project, I did 
# not adapt an existing codebase. I started out with a blank slate
# idea to classify music genres with a neural network. I made heavy 
# use of TensorFlow tutorials documentation and for ideas on the 
# structure and parameters of MLPs and CNNs for music genre 
# classification I have integrated code from Velardo’s YouTube 
# tutorial series (Velardo 2020).

# In-Code Citations will be inverted. I will cite where code was 
# taken from Velardo's tutorial series.

# Plan:
# create and train validation and test sets
# build the CNN
# compile the CNN
# train the CNN
# evaluate the CNN on the test set
# make a prediction!

import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt

dataset_path = "data_10.json"

def load_data(data_path):
    """Loads training dataset from json file.

        :param data_path (str): Path to json file containing data
        :return X (ndarray): Inputs
        :return y (ndarray): Targets
    """

    with open(data_path, "r") as fp:
        data = json.load(fp)

    X = np.array(data["mfcc"])
    y = np.array(data["labels"])
    return X, y

def prepare_datasets(test_size, validation_size):
    #load data
    X, y = load_data(dataset_path)
    
    #create train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    #create train/validation split
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)

    # make everything 3d for tensorflow with 1 channel (mfcc value)
    X_train = X_train[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_validation, X_test, y_train, y_validation, y_test
    
def predict(model, X, y):
    """Predict a single sample using the trained model

    :param model: Trained classifier
    :param X: Input data
    :param y (int): Target
    """

    # add a dimension to input data for sample - model.predict() expects a 4d array in this case
    X = X[np.newaxis, ...] # array shape (1, 130, 13, 1)

    # perform prediction
    prediction = model.predict(X)

    # get index with max value
    predicted_index = np.argmax(prediction, axis=1)

    print("Target: {}, Predicted label: {}".format(y, predicted_index))

if __name__ == "__main__":
    # create and train validation and test sets
    X_train, X_validation, X_test, y_train, y_validation, y_test  = prepare_datasets(0.25, 0.2)
    input_shape = (X_train.shape[1], X_train.shape[2], 1)

    # build the CNN
    model = keras.Sequential([

        # 1st layer
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'),
        keras.layers.BatchNormalization(),

        # 2nd layer
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'),
        keras.layers.BatchNormalization(),

        # 3rd layer
        keras.layers.Conv2D(32, (2, 2), activation='relu', input_shape=input_shape),
        keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'),
        keras.layers.BatchNormalization(),

        # output layer (softmax)
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(10, activation='softmax')
        

    ])

    # compile the CNN
    model.compile(optimizer='adam',
              loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    
    model.summary()

    # train the CNN    
    history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=32, epochs=30)

    # evaluate the CNN on the test set
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)

    # make a prediction!
    # pick a sample to predict from the test set
    X_to_predict = X_test[100]
    y_to_predict = y_test[100]

    # predict sample
    predict(model, X_to_predict, y_to_predict)

