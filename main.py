import pandas as pd
import matplotlib.pyplot as plt
# from xgboost import XGBClassifier
import numpy as np
from keras.src.layers import BatchNormalization
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout,Flatten
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
# it is important to estimate the model overall but also by dividing into segments
# especially since the dataset is imbalanced
from sklearn.metrics import classification_report, f1_score
from tensorflow.keras.optimizers import Adam
from keras.optimizers import Adagrad
# from xgboost import XGBClassifier
from tensorflow.keras.callbacks import EarlyStopping



random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)


# read the tracks from path of csv file . and presents 10 tracks in a table
def import_data(model = "model"):
    # read the tracks from path of csv file . and presents 10 tracks in a table
    pd.options.display.max_columns = 25
    if(model=="big_data_model"):
        path = "big_data_model.csv"
    else:
        path = "vehicle_tracks.csv"
    df = pd.read_csv(path)
    # Define the file path and name for the new file
    # Write the DataFrame to a new CSV file
    df.to_csv(path, index=False)
    print("print the first five rows from the tracks_database.csv file:")
    print(df.head(5))
    df = pd.DataFrame(df)
    return df


# Drawing a track by ID
# params: 1. pointer to data frame. 2.Define the vehicle id for which you want to draw the route
def drawing_track(df, vehicle_id):
    # Filter the dataframe to keep only the data for the selected vehicle
    df_vehicle = df[df['vehicle_id'] == vehicle_id]
    # Extract the local_x and local_y coordinates for the selected vehicle
    x = df_vehicle['local_x'].tolist()
    y = df_vehicle['local_y'].tolist()
    # Plot the route
    plt.plot(x,y)
    plt.xlabel("Local X")
    plt.ylabel("Local Y")
    plt.title(f"Route of Vehicle {vehicle_id}")
    plt.show()


#Normalize the data,
#The fit_transform() method is used to fit the data into a model and transform it into a form that is more suitable for the model in a single step.
def data_normalization(df_modified,model = "model"):
    # Initialize the scaler
    scaler = MinMaxScaler()

    # Normalize the local_x and local_y columns and add them as new columns in the DataFrame
    df_modified['local_x_norm'] = (df_modified['local_x'] - df_modified['local_x'].mean())/df_modified['local_x'].std()
    df_modified['local_y_norm'] = (df_modified['local_y'] - df_modified['local_y'].mean())/df_modified['local_y'].std()
    df_modified['global_time_norm'] = (df_modified['global_time'] - df_modified['global_time'].mean())/df_modified['global_time'].std()
    # df_modified['local_x_norm'] = scaler.fit_transform(df_modified[['local_x_norm']])
    # df_modified['local_y_norm'] = scaler.fit_transform(df_modified[['local_y_norm']])
    # Write the DataFrame to the same CSV file
    if(model == "big_data_model"):
        path = "big_tracks_database_modified.csv"
    else:
        path = "tracks_database_modified.csv"
    df_modified.to_csv(path, index=False)
    # Review the indicators.
    print(df_modified.describe())
    return df_modified


#Group by the cordinates to a tracks
def group_tracks(df_modified,model = "model"):
    if (model == "big_data_model"):
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
        "accident": int(x["accident"].all())
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

        # Loop over each route and store its data in a 3D array
        for x, y, time in zip(df_modified['local_x_norm'], df_modified['local_y_norm'], df_modified['global_time_norm']):
            route_data = list(zip(x, y, time))

            # Update the maximum route length if needed
            if len(route_data) > max_route_length:
                max_route_length = len(route_data)

            routes_data.append(route_data)

        # Pad sequences with tuples of zeros to ensure consistent length
        for i in range(len(routes_data)):
            route_data = routes_data[i]
            padded_data = [(0, 0, 0)] * (max_route_length - len(route_data)) + route_data
            routes_data[i] = padded_data


        # Convert the list of 3D arrays into a 3D NumPy array
        routes_data = np.array(routes_data)
        return routes_data,max_route_length


def read_y(df_modified):
    return df_modified['accident']


#Shows how many examples we have of each type to see if the database is balanced
def is_balanced_database(df_modified):
    print(df_modified['accident'].value_counts())

#=============================================================================================================================================
def oneHotEndcoding(Y_train, Y_test):
    Y_train_oh = pd.get_dummies(Y_train, prefix='type')
    Y_test_oh = pd.get_dummies(Y_test, prefix='type')
    return Y_train_oh, Y_test_oh


def callbacks_function(name):
  early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='min', restore_best_weights=True)
  monitor = tf.keras.callbacks.ModelCheckpoint(name, monitor='val_loss', verbose=0,save_best_only=False
                                               ,save_weights_only=False,mode='min')

  def scheduler(epoch, lr):
    if epoch%5 == 0 and epoch > 0:
      lr = lr/2
    return lr

  lr_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler,verbose = 0)

  return lr_schedule


def f1(y_true, y_pred):
  def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
    possible_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + tf.keras.backend.epsilon())
    return recall

  def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
    predicted_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
    return precision
  precision = precision(y_true, y_pred)
  recall = recall(y_true, y_pred)
  b=0.5
  return ((((1+b)**2)*(precision*recall))/(((b**2)*(precision))+recall+tf.keras.backend.epsilon()))

def model_comiple_run(num_epochs,initial_learning_rate,model,X_train,Y_train,X_test,y_test,callbacks,optimizer = "Adam"):
  #typeX_test - <class 'numpy.ndarray'>
  #type y_test - <class 'pandas.core.frame.DataFrame'>
  # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'precision', 'recall', 'f1_score'])

  # Define your desired initial learning rate
  # initial_learning_rate = 0.1  # Change this to your desired value
  # Create an optimizer with the specified learning rate
  opt = Adam(learning_rate=initial_learning_rate)
  if(optimizer == "Adagrade"):
      opt = Adagrad(learning_rate=initial_learning_rate)

  # Compile your model using the custom optimizer
  model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy', f1])

  # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy',f1])
  model_history = model.fit(X_train, Y_train,validation_data=(X_test, y_test), verbose=1, epochs= num_epochs)
  # batch_size=32, validation_data=(X_test,y_test)\,callbacks=callbacks,verbose=1)
  return model_history

# max_length = max(len(subarray[0]) for subarray in arr) # Find the maximum length of sub-arrays


#function definition for ploting the accuracy graph for train and validation sets.
def model_plot(model_history, type = None, plot_all = True):
  plt.plot()
  if plot_all:
    plt.plot(model_history.history['accuracy'], label = "Train accuracy")
    plt.plot(model_history.history['val_accuracy'], label = "Val accuracy")
    plt.xlabel("Epoch (iteration)")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.legend()
    plt.show()
    plt.plot(model_history.history['loss'], label = "Train loss")
    plt.plot(model_history.history['val_loss'], label = "Val loss")
    plt.xlabel("Epoch (iteration)")
    plt.ylabel("Loss")
    plt.grid()
    plt.legend()
    plt.show()
    plt.plot(model_history.history['f1'], label = "Train f1")
    plt.plot(model_history.history['val_f1'], label = "Val f1")
    plt.xlabel("Epoch (iteration)")
    plt.ylabel("F1")
    plt.grid()
    plt.legend()
    plt.show()
  else:
    if type=='accuracy':
      plt.plot(model_history.history['accuracy'], label = "Train accuracy")
      plt.plot(model_history.history['val_accuracy'], label = "val_accuracy")
    if type=='loss':
      plt.plot(model_history.history['loss'], label = "Train loss")
      plt.plot(model_history.history['val_loss'], label = "val_loss")
    if type=='f1':
      plt.plot(model_history.history['f1'], label = "Train f1")
      plt.plot(model_history.history['val_f1'], label = "val_f1")
    plt.legend()
    plt.show()
#=============================================================================================================================================

def encoder(x,y):
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
    print("sum(y)/len(y) = ", sum(labels_encoded)/len(labels_encoded))
    print("sum(y_train)/len(y_train)= ", sum(y_train)/len(y_train))
    print("sum(y_test)/len(y_test)= ", sum(y_test)/len(y_test))
    # y_train, y_test = oneHotEndcoding(y_train, y_test)
    return X_train, X_test, y_train, y_test



# LSTM - model 1
def lstm_model1(X_train, X_test, y_train, y_test,max_route_length):
    # Experiment 1: hidden_layers = 1, total_nodes = 64
    # Define the LSTM model
    model = Sequential()
    # input_shape - we define an LSTM model with an input shape of (param1, param2), meaning it takes in a sequence of param1 inputs with param2 feature each.
    model.add(LSTM(40, input_shape=(X_train.shape[1], X_train.shape[2])))
    # model.add(Dropout(0.2))
    model.add(Flatten())
    # The sigmoid activation function is commonly used for binary classification problems, where the output values range between 0 and 1
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    model_name = 'First'
    lr_schedule = callbacks_function(model_name)
    # Assuming X_train and X_test are your ragged nested sequences
    initial_learning_rate = 0.1
    num_epochs = 100
    model_history = model_comiple_run(num_epochs,initial_learning_rate,model, X_train, y_train, X_test, y_test,
                                      callbacks=[lr_schedule])
    model_plot(model_history)
    # model.load_weights(model_name)
    # loss = model.evaluate(X_test, y_test, verbose=1, steps=X_test.shape[0]) # Evaluate the model
    # print('Final loss 9 (cross-entropy and accuracy and F1):', loss)

    loss = model.evaluate(X_test, y_test, verbose=1, steps=X_test.shape[0]) # evaluate the model
    print('Final loss 1 (cross-entropy and accuracy and F1):', loss)

    y_pred = model.predict(X_test)
    threshold = 0.5   # Define a threshold for binary classification (e.g., 0.5)
    y_pred_binary = (y_pred >= threshold).astype(int)     # Convert predicted probabilities to binary labels
    print("y_test:\n", y_test.flatten())
    print("===========================================",y_pred_binary)
    print("===========================================", y_test)
    cf_matrix = confusion_matrix(y_test, y_pred_binary)
    plot_confusion_matrix(cf_matrix)

    # Calculate precision
    precision = precision_score(y_test, y_pred_binary)

    # Calculate recall
    recall = recall_score(y_test, y_pred_binary)

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    # # Classify new tracks using the trained model
    # new_track = [(102.4208282026001537,100.6556837362916167,1.7341556317074696),(4.4079540881162123, -2.4456476447699654, 1.734483396060324),(2.3953228814527225,-0.2356115532483141,1.7348111604131784),(2.3826916747892333,-0.0254120095153621,1.7351389247660327),(2.3700604681257427,0.1846240820062891,1.7354666891188872)]# Example new track values
    # new_track = [(0, 0, 0)] * (max_route_length - len(new_track)) + new_track
    # new_track = np.array(new_track)
    # new_track = new_track.reshape((1, new_track.shape[0], new_track.shape[1])) # Reshape to match LSTM input shape
    # # new_track = pad_sequences(new_track, maxlen=max_route_length)
    # prediction = model.predict(new_track)
    # # Decode the prediction
    # predicted_label = label_encoder.inverse_transform(np.round(prediction).astype(int))
    # # classification The new track
    # if prediction[0] >= 0.5:
    #     print("The new route is anomalous")
    # else:
    #     print("The new route is normal")



#LSTM - model 2
def lstm_model2(X_train, X_test, y_train, y_test,max_route_length):
    # Experiment 1: hidden_layers = 1, total_nodes = 64
    # Define the LSTM model
    model = Sequential()
    # input_shape - we define an LSTM model with an input shape of (param1, param2), meaning it takes in a sequence of param1 inputs with param2 feature each.
    model.add(LSTM(40, input_shape=(X_train.shape[1], X_train.shape[2])))
    # model.add(Dropout(0.2))
    model.add(Flatten())
    # The sigmoid activation function is commonly used for binary classification problems, where the output values range between 0 and 1
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    model_name = 'model2'
    lr_schedule = callbacks_function(model_name)
    # Assuming X_train and X_test are your ragged nested sequences
    initial_learning_rate = 0.001
    num_epochs =100
    model_history = model_comiple_run(num_epochs,initial_learning_rate,model, X_train, y_train, X_test, y_test,
                                      callbacks=[ lr_schedule])
    model_plot(model_history)
    # model.load_weights(model_name)
    # loss = model.evaluate(X_test, y_test, verbose=1, steps=X_test.shape[0]) # Evaluate the model
    # print('Final loss 9 (cross-entropy and accuracy and F1):', loss)

    loss = model.evaluate(X_test, y_test, verbose=1, steps=X_test.shape[0]) # evaluate the model
    print('Final loss 1 (cross-entropy and accuracy and F1):', loss)

    y_pred = model.predict(X_test)
    threshold = 0.5   # Define a threshold for binary classification (e.g., 0.5)
    y_pred_binary = (y_pred >= threshold).astype(int)     # Convert predicted probabilities to binary labels
    print("y_test:\n", y_test.flatten())
    print("===========================================",y_pred)
    print("===========================================", y_test)
    cf_matrix = confusion_matrix(y_test, y_pred_binary)
    plot_confusion_matrix(cf_matrix)

    # Calculate precision
    precision = precision_score(y_test, y_pred_binary)

    # Calculate recall
    recall = recall_score(y_test, y_pred_binary)

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")

#LSTM - model 3
def lstm_model3(X_train, X_test, y_train, y_test,max_route_length):
    tf.keras.backend.clear_session()
    # Experiment 1: hidden_layers = 1, total_nodes = 64
    # Define the LSTM model
    model = Sequential()
    # input_shape - we define an LSTM model with an input shape of (param1, param2), meaning it takes in a sequence of param1 inputs with param2 feature each.
    model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2])))
    # model.add(Dropout(0.2))
    model.add(Flatten())
    # The sigmoid activation function is commonly used for binary classification problems, where the output values range between 0 and 1
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    model_name = 'model3'
    lr_schedule = callbacks_function(model_name)
    # Assuming X_train and X_test are your ragged nested sequences
    initial_learning_rate = 0.001
    num_epochs =1000
    model_history = model_comiple_run(num_epochs,initial_learning_rate,model, X_train, y_train, X_test, y_test,
                                      callbacks=[ lr_schedule])
    model_plot(model_history)
    # model.load_weights(model_name)
    # loss = model.evaluate(X_test, y_test, verbose=1, steps=X_test.shape[0]) # Evaluate the model
    # print('Final loss 9 (cross-entropy and accuracy and F1):', loss)

    loss = model.evaluate(X_test, y_test, verbose=1, steps=X_test.shape[0]) # evaluate the model
    print('Final loss 1 (cross-entropy and accuracy and F1):', loss)

    y_pred = model.predict(X_test)
    threshold = 0.5   # Define a threshold for binary classification (e.g., 0.5)
    y_pred_binary = (y_pred >= threshold).astype(int)     # Convert predicted probabilities to binary labels
    print("y_test:\n", y_test.flatten())
    print("===========================================",y_pred)
    print("===========================================", y_test)
    cf_matrix = confusion_matrix(y_test, y_pred_binary)
    plot_confusion_matrix(cf_matrix)

    # Calculate precision
    precision = precision_score(y_test, y_pred_binary)

    # Calculate recall
    recall = recall_score(y_test, y_pred_binary)

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")


#LSTM - model 4
def lstm_model4(X_train, X_test, y_train, y_test,max_route_length):
    tf.keras.backend.clear_session()
    # Experiment 1: hidden_layers = 1, total_nodes = 64
    # Define the LSTM model
    model = Sequential()
    # input_shape - we define an LSTM model with an input shape of (param1, param2), meaning it takes in a sequence of param1 inputs with param2 feature each.
    model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2])))
    # model.add(Dropout(0.2))
    model.add(Flatten())
    # The sigmoid activation function is commonly used for binary classification problems, where the output values range between 0 and 1
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    model_name = 'model4'
    lr_schedule = callbacks_function(model_name)
    # Assuming X_train and X_test are your ragged nested sequences
    initial_learning_rate = 0.001
    num_epochs =100
    model_history = model_comiple_run(num_epochs,initial_learning_rate,model, X_train, y_train, X_test, y_test,
                                      callbacks=[ lr_schedule])
    model_plot(model_history)
    # model.load_weights(model_name)
    # loss = model.evaluate(X_test, y_test, verbose=1, steps=X_test.shape[0]) # Evaluate the model
    # print('Final loss 9 (cross-entropy and accuracy and F1):', loss)

    loss = model.evaluate(X_test, y_test, verbose=1, steps=X_test.shape[0]) # evaluate the model
    print('Final loss 1 (cross-entropy and accuracy and F1):', loss)

    y_pred = model.predict(X_test)
    threshold = 0.5   # Define a threshold for binary classification (e.g., 0.5)
    y_pred_binary = (y_pred >= threshold).astype(int)     # Convert predicted probabilities to binary labels
    print("y_test:\n", y_test.flatten())
    print("===========================================",y_pred)
    print("===========================================", y_test)
    cf_matrix = confusion_matrix(y_test, y_pred_binary)
    plot_confusion_matrix(cf_matrix)

    # Calculate precision
    precision = precision_score(y_test, y_pred_binary)

    # Calculate recall
    recall = recall_score(y_test, y_pred_binary)

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")




#LSTM - model 5
def lstm_model5(X_train, X_test, y_train, y_test,max_route_length):
    tf.keras.backend.clear_session()
    # Experiment 1: hidden_layers = 1, total_nodes = 64
    # Define the LSTM model
    model = Sequential()
    # input_shape - we define an LSTM model with an input shape of (param1, param2), meaning it takes in a sequence of param1 inputs with param2 feature each.
    model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]),return_sequences=True))
    model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2])))
    # model.add(Dropout(0.2))
    model.add(Flatten())
    # The sigmoid activation function is commonly used for binary classification problems, where the output values range between 0 and 1
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    model_name = 'model5'
    lr_schedule = callbacks_function(model_name)
    # Assuming X_train and X_test are your ragged nested sequences
    initial_learning_rate = 0.001
    num_epochs =100
    model_history = model_comiple_run(num_epochs,initial_learning_rate,model, X_train, y_train, X_test, y_test,
                                      callbacks=[ lr_schedule])
    model_plot(model_history)
    # model.load_weights(model_name)
    # loss = model.evaluate(X_test, y_test, verbose=1, steps=X_test.shape[0]) # Evaluate the model
    # print('Final loss 9 (cross-entropy and accuracy and F1):', loss)

    loss = model.evaluate(X_test, y_test, verbose=1, steps=X_test.shape[0]) # evaluate the model
    print('Final loss 1 (cross-entropy and accuracy and F1):', loss)

    y_pred = model.predict(X_test)
    threshold = 0.5   # Define a threshold for binary classification (e.g., 0.5)
    y_pred_binary = (y_pred >= threshold).astype(int)     # Convert predicted probabilities to binary labels
    print("y_test:\n", y_test.flatten())
    print("===========================================",y_pred)
    print("===========================================", y_test)
    cf_matrix = confusion_matrix(y_test, y_pred_binary)
    plot_confusion_matrix(cf_matrix)

    # Calculate precision
    precision = precision_score(y_test, y_pred_binary)

    # Calculate recall
    recall = recall_score(y_test, y_pred_binary)

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")



#LSTM - model 6
def lstm_model6(X_train, X_test, y_train, y_test,max_route_length):
    tf.keras.backend.clear_session()
    # Experiment 1: hidden_layers = 1, total_nodes = 64
    # Define the LSTM model
    model = Sequential()
    # input_shape - we define an LSTM model with an input shape of (param1, param2), meaning it takes in a sequence of param1 inputs with param2 feature each.
    model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]),return_sequences=True))
    model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2])))

    model.add(Flatten())
    # The sigmoid activation function is commonly used for binary classification problems, where the output values range between 0 and 1
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    model_name = 'model6'
    lr_schedule = callbacks_function(model_name)
    # Assuming X_train and X_test are your ragged nested sequences
    initial_learning_rate = 0.01
    num_epochs =100
    model_history = model_comiple_run( num_epochs,initial_learning_rate,model, X_train, y_train, X_test, y_test,
                                      callbacks=[ lr_schedule],optimizer="Adagrade")
    model_plot(model_history)
    # model.load_weights(model_name)
    # loss = model.evaluate(X_test, y_test, verbose=1, steps=X_test.shape[0]) # Evaluate the model
    # print('Final loss 9 (cross-entropy and accuracy and F1):', loss)

    loss = model.evaluate(X_test, y_test, verbose=1, steps=X_test.shape[0]) # evaluate the model
    print('Final loss 1 (cross-entropy and accuracy and F1):', loss)

    y_pred = model.predict(X_test)
    threshold = 0.5   # Define a threshold for binary classification (e.g., 0.5)
    y_pred_binary = (y_pred >= threshold).astype(int)     # Convert predicted probabilities to binary labels
    print("y_test:\n", y_test.flatten())
    print("===========================================",y_pred)
    print("===========================================", y_test)
    cf_matrix = confusion_matrix(y_test, y_pred_binary)
    plot_confusion_matrix(cf_matrix)

    # Calculate precision
    precision = precision_score(y_test, y_pred_binary)

    # Calculate recall
    recall = recall_score(y_test, y_pred_binary)

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")





#LSTM - model 7
def lstm_model7(X_train, X_test, y_train, y_test,max_route_length):
    tf.keras.backend.clear_session()
    # Experiment 1: hidden_layers = 1, total_nodes = 64
    # Define the LSTM model
    model = Sequential()
    # input_shape - we define an LSTM model with an input shape of (param1, param2), meaning it takes in a sequence of param1 inputs with param2 feature each.
    model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]),return_sequences=True))
    model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2])))
    # model.add(Dropout(0.2))
    model.add(Flatten())
    # The sigmoid activation function is commonly used for binary classification problems, where the output values range between 0 and 1
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    model_name = 'model7'
    lr_schedule = callbacks_function(model_name)
    # Assuming X_train and X_test are your ragged nested sequences
    initial_learning_rate = 0.001
    num_epochs =100
    model_history = model_comiple_run(num_epochs,initial_learning_rate,model, X_train, y_train, X_test, y_test,
                                      callbacks=[ lr_schedule])
    model_plot(model_history)
    # model.load_weights(model_name)
    # loss = model.evaluate(X_test, y_test, verbose=1, steps=X_test.shape[0]) # Evaluate the model
    # print('Final loss 9 (cross-entropy and accuracy and F1):', loss)

    loss = model.evaluate(X_test, y_test, verbose=1, steps=X_test.shape[0]) # evaluate the model
    print('Final loss 1 (cross-entropy and accuracy and F1):', loss)

    y_pred = model.predict(X_test)
    threshold = 0.5   # Define a threshold for binary classification (e.g., 0.5)
    y_pred_binary = (y_pred >= threshold).astype(int)     # Convert predicted probabilities to binary labels
    print("y_test:\n", y_test.flatten())
    print("===========================================",y_pred)
    print("===========================================", y_test)
    cf_matrix = confusion_matrix(y_test, y_pred_binary)
    plot_confusion_matrix(cf_matrix)

    # Calculate precision
    precision = precision_score(y_test, y_pred_binary)

    # Calculate recall
    recall = recall_score(y_test, y_pred_binary)

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")



def calculate_next_point(last_x, last_y, angle, distance):
    c = last_x + 10
    d = last_y + 10
    while True:
        x = last_x + distance * math.cos(angle)
        y = last_y + distance * math.sin(angle)
        # Check distance condition
        if math.sqrt((x - c)**2 + (y - d)**2) <= 5:
            break
        else:
            angle += math.pi / 180  # Increase angle by 1 degree and retry

    return x, y

def calculate_accident_point(last_x, last_y, angle, distance):
    c = last_x + 10
    d = last_y + 10
    while True:
        x = last_x + distance * math.cos(angle)
        y = last_y + distance * math.sin(angle)
        # Check distance condition
        if (math.sqrt((x - c)**2 + (y - d)**2) > 11) and (y > last_y):
            break
        else:
            angle += math.radians(5)  # Increase angle by 5 degree and retry

    return x, y

def generate_route_points(num_points, distance_increment, noisy_points, has_accident, vehicle_id):
    route_points = []
    current_time = int(time.time())

    for i in range(num_points):
        x = y = i * distance_increment
        global_time = current_time + i * 10  # Increase timestamp by 10 seconds for each point
        accident_value = 1 if i in noisy_points else 0  # Set accident column value based on noisy points
        route_points.append((x, y, global_time, accident_value, vehicle_id))  # Include vehicle_id column

    accident_point = random.choice(noisy_points) if has_accident else -1
    for random_index in noisy_points:
        last_x, last_y, _, _, _ = route_points[random_index - 1]  # Ignore vehicle_id column
        angle = random.uniform(0, math.pi / 4)  # Limit to 45-degree angle

        # Calculate the random noisy point with a 10-meter distance
        noisy_x, noisy_y = calculate_next_point(last_x, last_y, angle, distance_increment)
        route_points.insert(random_index, (noisy_x, noisy_y, current_time + random_index * 10,0, vehicle_id))

        # Remove the next point after the noisy point
        route_points.pop(random_index + 1)

        # Introduce accident in vehicles with 50% probability
        if has_accident and random_index == accident_point:
            accident_x, accident_y = calculate_accident_point(last_x, last_y, angle, distance_increment)
            route_points[accident_point] = (accident_x, accident_y, current_time + accident_point * 10, 1, vehicle_id)
            print (vehicle_id)

    return route_points

def write_to_csv(filename, data):
    with open(filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['local_x', 'local_y', 'global_time', 'accident', 'vehicle_id'])
        for row in data:
            csv_writer.writerow(row)


def creating_synthetic_track_database():
    num_points = 36  # Number of points along the curve
    distance_increment = 10  # Increment in meters between points
    num_noisy_points = 20  # Number of noisy points

    # Choose 7 random indices for noisy points
    noisy_points = random.sample(range(1, num_points - 1), num_noisy_points)

    combined_route_points = []

    vehicle_data = {}  # Dictionary to store vehicle data
    for vehicle_id in range(1,1500):  # Create 5 different vehicle tracks
        has_accident = random.random() < 0.2  # 50% probability of having an accident
        route_points = generate_route_points(num_points, distance_increment, noisy_points, has_accident, f'vehicle_{vehicle_id}')
        combined_route_points.extend(route_points)

        vehicle_data[f'vehicle_{vehicle_id}'] = has_accident
    # Check if a vehicle has an accident, then update all rows for that vehicle
    for vehicle_id, has_accident in vehicle_data.items():
        if has_accident:
            for i in range(len(combined_route_points)):
                if combined_route_points[i][4] == vehicle_id:
                    combined_route_points[i] = (combined_route_points[i][0], combined_route_points[i][1],
                                               combined_route_points[i][2], 1, vehicle_id)
    write_to_csv('vehicle_tracks.csv', combined_route_points)


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


def xgboost_model(X_train, X_test, y_train, y_test):
    model = XGBClassifier(n_estimators=100,objective='binary:logistic', missing=1, seed=42,subsample=0.5, learning_rate=0.1, max_depth=4,eval_metric='aucpr', early_stopping_rounds=10,)
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    model.fit(X_train, y_train,verbose=True,
                    eval_set=[(X_test,y_test)])
    print(model)
    y_pred = model.predict(X_test)
    print("y_test:\n",y_test.flatten())
    print("y_pred:\n",y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    cf_matrix = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cf_matrix)
    print(classification_report(y_test, y_pred))


if __name__ == '__main__':
    # creating_synthetic_track_database()
    data_frame = import_data()
    vehicle = "vehicle_1099"
    drawing_track(data_frame, vehicle)
    df_modified = data_normalization(data_frame)
    df_modified = group_tracks(df_modified)
    x,max_route_length = reshape_tracks(df_modified)
    y = read_y(df_modified)
    is_balanced_database(df_modified)
    X_train, X_test, y_train, y_test = encoder(x,y)
    # lstm_model1(X_train, X_test, y_train, y_test,max_route_length)
    # lstm_model2(X_train, X_test, y_train, y_test,max_route_length)
    # lstm_model3(X_train, X_test, y_train, y_test,max_route_length)
    # increase the number of data.
    # data_frame = import_data("big_data_model")
    # vehicle = "vehicle_1099"
    # drawing_track(data_frame, vehicle)
    # df_modified = data_normalization(data_frame,"big_data_model")
    # df_modified = group_tracks(df_modified,"big_data_model")
    # x, max_route_length = reshape_tracks(df_modified)
    # y = read_y(df_modified)
    # is_balanced_database(df_modified)
    # X_train, X_test, y_train, y_test = encoder(x, y)
    # lstm_model4(X_train, X_test, y_train, y_test,max_route_length)
    # lstm_model5(X_train, X_test, y_train, y_test,max_route_length)
    # lstm_model6(X_train, X_test, y_train, y_test,max_route_length)
    lstm_model7(X_train, X_test, y_train, y_test,max_route_length)

    # xgboost_model(X_train, X_test, y_train, y_test)
