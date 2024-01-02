import os
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.src.layers import BatchNormalization
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from sklearn.model_selection import train_test_split
import ast
import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Flatten
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score
import tensorflow as tf
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import csv
import time
import random
import math
import seaborn as sns
from sklearn.metrics import classification_report, f1_score
from tensorflow.keras.optimizers import Adam
from keras.optimizers import Adagrad
from tensorflow.keras.callbacks import EarlyStopping

# from main import plot_confusion_matrix


#=================================================================================================================================
def callbacks_function(name):
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='min',
                                                  restore_best_weights=True)
    monitor = tf.keras.callbacks.ModelCheckpoint(name, monitor='val_loss', verbose=0, save_best_only=False
                                                 , save_weights_only=False, mode='min')

    # This functuon decrease the value of learning rate
    def scheduler(epoch, lr):
        if epoch % 5 == 0 and epoch > 0:
            lr = lr / 2
        return lr

    lr_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=0)

    return lr_schedule

    lr_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=0)
    return early_stop, monitor, lr_schedule


# This function calculates the f1 values
def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
        possible_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + tf.keras.backend.epsilon())
        return recall

    # This function calculates the precision
    def precision(y_true, y_pred):
        true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
        predicted_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    b = 0.5
    return ((((1 + b) ** 2) * (precision * recall)) / (((b ** 2) * (precision)) + recall + tf.keras.backend.epsilon()))


# This function uses to compile the models
def model_comiple_run(num_epochs, initial_learning_rate, model, X_train, Y_train, X_test, y_test, callbacks,
                      optimizer="Adam"):
    opt = Adam(learning_rate=initial_learning_rate)
    if (optimizer == "Adagrade"):
        opt = Adagrad(learning_rate=initial_learning_rate)

    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy', f1])
    model_history = model.fit(X_train, Y_train, validation_data=(X_test, y_test), verbose=1, epochs=num_epochs)
    # batch_size=32, validation_data=(X_test,y_test)\,callbacks=callbacks,verbose=1)
    return model_history


# function definition for ploting the accuracy graph for train and validation sets.
def model_plot(model_history, type=None, plot_all=True):
    plt.plot()
    if plot_all:
        plt.plot(model_history.history['accuracy'], label="Train accuracy")
        plt.plot(model_history.history['val_accuracy'], label="Val accuracy")
        plt.xlabel("Epoch (iteration)")
        plt.ylabel("Accuracy")
        plt.grid()
        plt.legend()
        plt.show()
        plt.plot(model_history.history['loss'], label="Train loss")
        plt.plot(model_history.history['val_loss'], label="Val loss")
        plt.xlabel("Epoch (iteration)")
        plt.ylabel("Loss")
        plt.grid()
        plt.legend()
        plt.show()
        plt.plot(model_history.history['f1'], label="Train f1")
        plt.plot(model_history.history['val_f1'], label="Val f1")
        plt.xlabel("Epoch (iteration)")
        plt.ylabel("F1")
        plt.grid()
        plt.legend()
        plt.show()
    else:
        if type == 'accuracy':
            plt.plot(model_history.history['accuracy'], label="Train accuracy")
            plt.plot(model_history.history['val_accuracy'], label="val_accuracy")
        if type == 'loss':
            plt.plot(model_history.history['loss'], label="Train loss")
            plt.plot(model_history.history['val_loss'], label="val_loss")
        if type == 'f1':
            plt.plot(model_history.history['f1'], label="Train f1")
            plt.plot(model_history.history['val_f1'], label="val_f1")
        plt.legend()
        plt.show()


def encoder(x, y):
  # Encode the target variable
  label_encoder = LabelEncoder()
  labels_encoded = label_encoder.fit_transform(y)
  labels_encoded = labels_encoded.reshape((len(y), 1))
  # Pad sequences to ensure consistent length
  # test_size - determines the percentage division into test and train.
  # random_state - Selects regular examples for training and testing in all running.
  # stratify = y - We will make sure that the training and test sets maintain the original ratio.
  X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                        random_state=42, stratify=y, shuffle = True)
  return X_train, X_test, y_train, y_test
#================================================================================================================================

#Upload database
def upload_database():
    # Assuming you have all CSV files in a directory named 'dataset'
    data_dir = "dataset"
    files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

    # Create an empty list to store data from each CSV file
    X_list = []
    Y_list = []

    # Load each CSV file into a separate DataFrame and append it to the list
    for file in files:
        df = pd.read_csv(os.path.join(data_dir, file))
        X_list.append(df.drop(columns=['Lable']))
        # Y_list.append(df['Lable']) # insert Y into 2 dimation array
        Y_list.append(df['Lable'][0])

    # Convert the list of arrays to a single NumPy array
    X = np.array(X_list)
    Y = np.array(Y_list)

    print("================",Y)
    print("================",len(Y))

    return X,Y

def lstm_model1(X_train, X_test, y_train, y_test):
    model = Sequential()
    # input_shape - we define an LSTM model with an input shape of (param1, param2), meaning it takes in a sequence
    # of param1 inputs with param2 feature each.
    model.add(LSTM(40, input_shape=(X_train.shape[1], X_train.shape[2])))
    # model.add(Dropout(0.2))
    model.add(Flatten())
    # The sigmoid activation function is commonly used for binary classification problems, where the output values
    # range between 0 and 1
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    model_name = 'Model1'
    lr_schedule = callbacks_function(model_name)
    initial_learning_rate = 0.1
    num_epochs = 100
    model_history = model_comiple_run(num_epochs, initial_learning_rate, model, X_train, y_train, X_test, y_test,
                                      callbacks=[lr_schedule])
    model_plot(model_history)
    loss = model.evaluate(X_test, y_test, verbose=1, steps=X_test.shape[0])  # evaluate the model
    print('Final loss 1 (cross-entropy and accuracy and F1):', loss)

    y_pred = model.predict(X_test)
    # Define a threshold for binary classification (e.g., 0.5)
    threshold = 0.5
    y_pred_binary = (y_pred >= threshold).astype(int)  # Convert predicted probabilities to binary labels
    cf_matrix = confusion_matrix(y_test, y_pred_binary)
    plot_confusion_matrix(cf_matrix)
    # Calculate precision
    precision = precision_score(y_test, y_pred_binary)
    # Calculate recall
    recall = recall_score(y_test, y_pred_binary)
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")


# Define a custom normalization function
def data_normalization(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    normalized_data = (data - mean) / (std + 1e-8)  # Add a small epsilon to avoid division by zero
    return normalized_data


if __name__ == '__main__':
    X,Y = upload_database()
    X_normalized = data_normalization(X)
# Inside the 'encoder' function
    print("Mean of Normalized Data:")
    print(np.mean(X_normalized))  # Print the mean along each column
    print("Standard Deviation of Normalized Data:")
    print(np.std(X_normalized))  # Print the standard deviation along each column
    X_train, X_test, y_train, y_test = encoder(X_normalized, Y)
    lstm_model1(X_train, X_test, y_train, y_test)