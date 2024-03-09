import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.src.layers import BatchNormalization
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
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

random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)


# read the tracks from path of csv file  and presents 10 tracks in a table
def import_data(model="model"):
    # read the tracks from path of csv file . and presents 10 tracks in a table
    pd.options.display.max_columns = 25
    if model == "big_data_model":
        path = "big_vehicle_tracks_database.csv"
    else:
        path = "small_vehicle_tracks_database.csv"
    df = pd.read_csv(path)
    # Define the file path and name for the new file
    # Write the DataFrame to a new CSV file
    df.to_csv(path, index=False)
    print("print the first five rows from the tracks_database.csv file:")
    print(df.head(5))
    df = pd.DataFrame(df)
    return df


# Normalize the data according to the equation we learned in the class
def data_normalization(df_modified, model="model"):
    # Normalize the local_x and local_y columns and add them as new columns in the DataFrame
    df_modified['local_x_norm'] = (df_modified['local_x'] - df_modified['local_x'].mean()) / df_modified[
        'local_x'].std()
    df_modified['local_y_norm'] = (df_modified['local_y'] - df_modified['local_y'].mean()) / df_modified[
        'local_y'].std()
    df_modified['global_time_norm'] = (df_modified['global_time'] - df_modified['global_time'].mean()) / df_modified[
        'global_time'].std()
    df_modified['velocity_norm'] = (df_modified['velocity'] - df_modified['velocity'].mean()) / df_modified[
        'velocity'].std()
    df_modified['acceleration_norm'] = (df_modified['acceleration'] - df_modified['acceleration'].mean()) / df_modified[
        'acceleration'].std()
    # Write the DataFrame to the same CSV file
    if model == "big_data_model":
        path = "big_tracks_database_modified.csv"
    else:
        path = "tracks_database_modified.csv"
    df_modified.to_csv(path, index=False)
    # Review the indicators.
    print(df_modified.describe())
    return df_modified


# Group by the cordinates to a tracks
def group_tracks(df_modified, model="model"):
    if model == "big_data_model":
        path = "big_tracks_database_modified.csv"
    else:
        path = "tracks_database_modified.csv"
    # Load data from CSV file
    df_modified = pd.read_csv(path)
    # Sort the DataFrame by 'global_time' to preserve the original order
    sorted_df = df_modified.sort_values('global_time')

    # Group by 'vehicle_id' and preserve the original order by using 'first' aggregation function
    grouped = sorted_df.groupby("vehicle_id").apply(lambda x: pd.Series({
        "vehicle_id": x["vehicle_id"].iloc[0],  # Add vehicle_id column
        "local_x_norm": np.array(x["local_x_norm"]).astype(float).tolist(),
        "local_y_norm": x["local_y_norm"].astype(float).tolist(),
        "global_time_norm": x["global_time_norm"].astype(float).tolist(),
        "accident": int(x["accident"].all()),
        "velocity_norm": x["velocity_norm"].astype(float).tolist(),
        "acceleration_norm": x["acceleration_norm"].astype(float).tolist(),
    }))

    # Reset the index to remove the multi-level index created by groupby
    grouped.reset_index(drop=True, inplace=True)

    # Save grouped data to CSV file
    grouped.to_csv(path, index=False)
    return grouped


def reshape_tracks(df_modified):
    # Create a list to store the 3D array for each route
    routes_data = []

    # Define the maximum route length
    max_route_length = 0

    # Loop over each route and store its data in a 5D array
    for x, y, time,velocity,acceleration in zip(df_modified['local_x_norm'], df_modified['local_y_norm'], df_modified['global_time_norm'],df_modified['velocity_norm'],df_modified['acceleration_norm']):
        route_data = list(zip(x, y, time,velocity,acceleration))

        # Update the maximum route length if needed
        if len(route_data) > max_route_length:
            max_route_length = len(route_data)

        routes_data.append(route_data)

    # Pad sequences with tuples of zeros to ensure consistent length
    for i in range(len(routes_data)):
        route_data = routes_data[i]
        padded_data = [(0, 0, 0, 0, 0)] * (max_route_length - len(route_data)) + route_data
        routes_data[i] = padded_data

    # Convert the list of 5D arrays into a 5D NumPy array
    routes_data = np.array(routes_data)
    return routes_data, max_route_length


def read_y(df_modified):
    return df_modified['accident']


# Shows how many examples we have of each type to see if the database is balanced
def is_balanced_database(df_modified):
    print(df_modified['accident'].value_counts())


# =======================================================================================================

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


# =============================================================================================================================

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


# ====================================================================================================================

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


# =============================================================================================================================================

def encoder(x, y):
    # Encode the target variable
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(y)
    labels_encoded = labels_encoded.reshape((len(y), 1))
    # Pad sequences to ensure consistent length
    # test_size - determines the percentage division into test and train.
    # random_state - Selects regular examples for training and testing in all running.
    # stratify = y - We will make sure that the training and test sets maintain the original ratio.
    X_train, X_test, y_train, y_test = train_test_split(x, labels_encoded, test_size=0.2,
                                                        random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test


# ====================================================================================================================

# LSTM - model 1 : hidden_layers = 1, total_nodes = 40, initial_learning_rate = 0.1, num_epochs = 100, with small
# DataBase - "tracks_database.csv",optimizer='adam'
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


# ======================================================================================================================

# LSTM - model 2 : hidden_layers = 1, total_nodes = 40, initial_learning_rate = 0.001, num_epochs = 100, with small
# DataBase - "tracks_database.csv",optimizer='adam'
# Improvement : initial_learning_rate = 0.001
def lstm_model2(X_train, X_test, y_train, y_test):
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
    model_name = 'Model2'
    lr_schedule = callbacks_function(model_name)
    initial_learning_rate = 0.001
    num_epochs = 100
    model_history = model_comiple_run(num_epochs, initial_learning_rate, model, X_train, y_train, X_test, y_test,
                                      callbacks=[lr_schedule])
    model_plot(model_history)
    loss = model.evaluate(X_test, y_test, verbose=1, steps=X_test.shape[0])  # evaluate the model
    print('Final loss 1 (cross-entropy and accuracy and F1):', loss)
    y_pred = model.predict(X_test)
    threshold = 0.5  # Define a threshold for binary classification (e.g., 0.5)
    y_pred_binary = (y_pred >= threshold).astype(int)  # Convert predicted probabilities to binary labels
    cf_matrix = confusion_matrix(y_test, y_pred_binary)
    plot_confusion_matrix(cf_matrix)
    # Calculate precision
    precision = precision_score(y_test, y_pred_binary)
    # Calculate recall
    recall = recall_score(y_test, y_pred_binary)
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")


# ===============================================================================================================

# LSTM - model 3 : hidden_layers = 1, total_nodes = 128, initial_learning_rate = 0.001, num_epochs = 1000,
# with small DataBase - "tracks_database.csv",optimizer='adam'
# Improvement : total_nodes = 128,num_epochs = 1000
def lstm_model3(X_train, X_test, y_train, y_test):
    tf.keras.backend.clear_session()
    model = Sequential()
    # input_shape - we define an LSTM model with an input shape of (param1, param2), meaning it takes in a sequence
    # of param1 inputs with param2 feature each.
    model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Flatten())
    # The sigmoid activation function is commonly used for binary classification problems, where the output values
    # range between 0 and 1
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    model_name = 'Model3'
    lr_schedule = callbacks_function(model_name)
    initial_learning_rate = 0.001
    num_epochs = 1000
    model_history = model_comiple_run(num_epochs, initial_learning_rate, model, X_train, y_train, X_test, y_test,
                                      callbacks=[lr_schedule])
    model_plot(model_history)
    loss = model.evaluate(X_test, y_test, verbose=1, steps=X_test.shape[0])  # evaluate the model
    print('Final loss 1 (cross-entropy and accuracy and F1):', loss)
    y_pred = model.predict(X_test)
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


# =========================================================================================================

# LSTM - model 4 : hidden_layers = 1, total_nodes = 128, initial_learning_rate = 0.001, num_epochs = 100, DataBase -
# the big DB - "big_data_model.csv", optimizer='adam'
# Improvement : DataBase - the big DB - "big_data_model.csv", num_epochs = 100
def lstm_model4(X_train, X_test, y_train, y_test):
    tf.keras.backend.clear_session()
    model = Sequential()
    # input_shape - we define an LSTM model with an input shape of (param1, param2), meaning it takes in a sequence
    # of param1 inputs with param2 feature each.
    model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2])))
    # model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    model_name = 'Model4'
    lr_schedule = callbacks_function(model_name)
    initial_learning_rate = 0.001
    num_epochs = 100
    model_history = model_comiple_run(num_epochs, initial_learning_rate, model, X_train, y_train, X_test, y_test,
                                      callbacks=[lr_schedule])
    model_plot(model_history)
    loss = model.evaluate(X_test, y_test, verbose=1, steps=X_test.shape[0])  # evaluate the model
    print('Final loss 1 (cross-entropy and accuracy and F1):', loss)
    y_pred = model.predict(X_test)
    threshold = 0.5  # Define a threshold for binary classification (e.g., 0.5)
    y_pred_binary = (y_pred >= threshold).astype(int)  # Convert predicted probabilities to binary labels
    cf_matrix = confusion_matrix(y_test, y_pred_binary)
    plot_confusion_matrix(cf_matrix)
    # Calculate precision & recall
    precision = precision_score(y_test, y_pred_binary)
    recall = recall_score(y_test, y_pred_binary)
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")


# ========================================================================================================
# LSTM - model 5: hidden_layers = 2, total_nodes = 128, initial_learning_rate = 0.001, num_epochs = 100, DataBase -
# the big DB - "big_data_model.csv", optimizer='adam'
# Improvement : hidden_layers = 2 (Adding one more lstm layer)
def lstm_model5(X_train, X_test, y_train, y_test):
    tf.keras.backend.clear_session()
    model = Sequential()
    # input_shape - we define an LSTM model with an input shape of (param1, param2), meaning it takes in a sequence
    # of param1 inputs with param2 feature each.
    model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
    model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2])))
    # model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    model_name = 'Model5'
    lr_schedule = callbacks_function(model_name)
    initial_learning_rate = 0.001
    num_epochs = 100
    model_history = model_comiple_run(num_epochs, initial_learning_rate, model, X_train, y_train, X_test, y_test,
                                      callbacks=[lr_schedule])
    model_plot(model_history)
    loss = model.evaluate(X_test, y_test, verbose=1, steps=X_test.shape[0])  # evaluate the model
    print('Final loss 1 (cross-entropy and accuracy and F1):', loss)

    y_pred = model.predict(X_test)
    threshold = 0.5  # Define a threshold for binary classification (e.g., 0.5)
    y_pred_binary = (y_pred >= threshold).astype(int)  # Convert predicted probabilities to binary labels
    cf_matrix = confusion_matrix(y_test, y_pred_binary)
    plot_confusion_matrix(cf_matrix)
    # Calculate precision & recall
    precision = precision_score(y_test, y_pred_binary)
    recall = recall_score(y_test, y_pred_binary)
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")


# ===================================================================================================================

# LSTM - model 6: hidden_layers = 2, total_nodes = 128, initial_learning_rate = 0.001, num_epochs = 100, DataBase -
# the big DB - "big_data_model.csv", optimizer='Adagrade'
# Improvement : optimizer='Adagrade'
def lstm_model6(X_train, X_test, y_train, y_test):
    tf.keras.backend.clear_session()
    model = Sequential()
    # input_shape - we define an LSTM model with an input shape of (param1, param2), meaning it takes in a sequence
    # of param1 inputs with param2 feature each.
    model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
    model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    model_name = 'Model6'
    lr_schedule = callbacks_function(model_name)
    initial_learning_rate = 0.001
    num_epochs = 100
    model_history = model_comiple_run(num_epochs, initial_learning_rate, model, X_train, y_train, X_test, y_test,
                                      callbacks=[lr_schedule], optimizer="Adagrade")
    model_plot(model_history)
    loss = model.evaluate(X_test, y_test, verbose=1, steps=X_test.shape[0])  # evaluate the model
    print('Final loss 1 (cross-entropy and accuracy and F1):', loss)
    y_pred = model.predict(X_test)
    threshold = 0.5  # Define a threshold for binary classification (e.g., 0.5)
    y_pred_binary = (y_pred >= threshold).astype(int)  # Convert predicted probabilities to binary labels
    cf_matrix = confusion_matrix(y_test, y_pred_binary)
    plot_confusion_matrix(cf_matrix)
    # Calculate precision & recall
    precision = precision_score(y_test, y_pred_binary)
    recall = recall_score(y_test, y_pred_binary)
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")


# =============================================================================================================

# LSTM - model 7: hidden_layers = 2, total_nodes = 128, initial_learning_rate = 0.01, num_epochs = 100, DataBase -
# the big DB - "big_data_model.csv", optimizer='Adagrade'
# Improvement : optimizer='Adagrade', initial_learning_rate = 0.01
def lstm_model7(X_train, X_test, y_train, y_test):
    tf.keras.backend.clear_session()
    model = Sequential()
    # input_shape - we define an LSTM model with an input shape of (param1, param2), meaning it takes in a sequence
    # of param1 inputs with param2 feature each.
    model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
    model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    model_name = 'Model7'
    lr_schedule = callbacks_function(model_name)
    initial_learning_rate = 0.01
    num_epochs = 100
    model_history = model_comiple_run(num_epochs, initial_learning_rate, model, X_train, y_train, X_test, y_test,
                                      callbacks=[lr_schedule], optimizer="Adagrade")
    model_plot(model_history)
    loss = model.evaluate(X_test, y_test, verbose=1, steps=X_test.shape[0])  # evaluate the model
    print('Final loss 1 (cross-entropy and accuracy and F1):', loss)
    y_pred = model.predict(X_test)
    threshold = 0.5
    y_pred_binary = (y_pred >= threshold).astype(int)  # Convert predicted probabilities to binary labels
    cf_matrix = confusion_matrix(y_test, y_pred_binary)
    plot_confusion_matrix(cf_matrix)
    # Calculate precision & recall
    precision = precision_score(y_test, y_pred_binary)
    recall = recall_score(y_test, y_pred_binary)
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")


# ==============================================================================================================

# LSTM - model 8: hidden_layers = 2, total_nodes = 128, initial_learning_rate = 0.001, num_epochs = 100, DataBase -
# the big DB - "big_data_model.csv", optimizer='adam'
# Improvement : threshold = 0.3
def lstm_model8(X_train, X_test, y_train, y_test):
    tf.keras.backend.clear_session()
    model = Sequential()
    # input_shape - we define an LSTM model with an input shape of (param1, param2), meaning it takes in a sequence of param1 inputs with param2 feature each.
    model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
    model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    model_name = 'Model8'
    lr_schedule = callbacks_function(model_name)
    initial_learning_rate = 0.001
    num_epochs = 100
    model_history = model_comiple_run(num_epochs, initial_learning_rate, model, X_train, y_train, X_test, y_test,
                                      callbacks=[lr_schedule])
    model_plot(model_history)
    loss = model.evaluate(X_test, y_test, verbose=1, steps=X_test.shape[0])  # evaluate the model
    print('Final loss 1 (cross-entropy and accuracy and F1):', loss)
    y_pred = model.predict(X_test)
    threshold = 0.3  # Define a threshold for binary classification (e.g., 0.5)
    y_pred_binary = (y_pred >= threshold).astype(int)  # Convert predicted probabilities to binary labels
    cf_matrix = confusion_matrix(y_test, y_pred_binary)
    plot_confusion_matrix(cf_matrix)
    # Calculate precision & recall
    precision = precision_score(y_test, y_pred_binary)
    recall = recall_score(y_test, y_pred_binary)
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")


# ===============================================================

def plot_confusion_matrix(cf_matrix):
    print('Confusion Matrix')
    ax = plt.subplot()
    sns.heatmap(cf_matrix, annot=True, ax=ax, fmt='d', cmap='Blues', cbar=False)
    # labels, title and ticks
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('Actual labels')
    ax.xaxis.set_ticklabels(['Not Accident', 'Accident'])
    ax.yaxis.set_ticklabels(['Not accident', 'Accident'])
    plt.show()  # Display the plot




def plot_confusion_matrix(cf_matrix):
    print('Confusion Matrix')
    ax = plt.subplot()
    sns.heatmap(cf_matrix, annot=True, ax=ax, fmt='d', cmap='Blues', cbar=False)
    # labels, title and ticks
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('Actual labels')
    ax.xaxis.set_ticklabels(['Not accident', 'Accident'])
    ax.yaxis.set_ticklabels(['Not accident', 'Accident'])
    plt.show()  # Display the plot

if __name__ == '__main__':
    # Data preparation
    data_frame = import_data()
    df_modified = data_normalization(data_frame)
    df_modified = group_tracks(df_modified)
    x, max_route_length = reshape_tracks(df_modified)
    y = read_y(df_modified)
    is_balanced_database(df_modified)
    X_train, X_test, y_train, y_test = encoder(x, y)
    # Running the models
    # Base model
    lstm_model1(X_train, X_test, y_train, y_test)
    # improvements
    lstm_model2(X_train, X_test, y_train, y_test)
    lstm_model3(X_train, X_test, y_train, y_test)
    # increase the number of data.
    data_frame = import_data("big_data_model")
    df_modified = data_normalization(data_frame, "big_data_model")
    df_modified = group_tracks(df_modified, "big_data_model")
    x, max_route_length = reshape_tracks(df_modified)
    y = read_y(df_modified)
    is_balanced_database(df_modified)
    X_train, X_test, y_train, y_test = encoder(x, y)
    lstm_model4(X_train, X_test, y_train, y_test)
    lstm_model5(X_train, X_test, y_train, y_test)
    lstm_model6(X_train, X_test, y_train, y_test)
    lstm_model7(X_train, X_test, y_train, y_test)
    lstm_model8(X_train, X_test, y_train, y_test)