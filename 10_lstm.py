import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
import tensorflow as tf


class Data:
    
    def __init__(self, source_file_name):
        #read csv or excel data
        if source_file_name[-3:] == 'csv':
            df = pd.read_csv(source_file_name, parse_dates = ['timestamp'])
        else:
            df = pd.read_excel(source_file_name, parse_dates = ['timestamp'])

        #set timestamp as index
        df.set_index('timestamp', inplace = True)
        
        #mark stable as 1
        df['stable'] = df['stable'].fillna(1)

        #set input and output
        X_stabil = df['count']
        y_stabil = df['stable']

        self.X_stable = X_stable
        self.y_stable = y_stable
        
    def splitTrainTest(self, input_test_size):
        #split data to train and test
        X_train, X_test, y_train, y_test = train_test_split(self.X_stable, self.y_stable, test_size = float(input_test_size), random_state = 42, shuffle = False)
        X_train = X_train.to_numpy()
        X_test = X_test.to_numpy()
        y_train = y_train.to_numpy()
        y_test = y_test.to_numpy()
        
        #reshape input for lstm (batch_size, timesteps, features)
        X_train = X_train.reshape((X_train.shape[0],1,1))
        X_test = X_test.reshape((X_test.shape[0],1,1))

        return X_train, X_test, y_train, y_test

class Model:

    def __init__(self, number_units, number_epochs, number_batch_size, number_dropout):
        self.number_units = number_units
        self.number_epochs = number_epochs
        self.number_batch_size = number_batch_size
        self.number_dropout = float(number_dropout)

    def train(self):
        #get data
        X_train, X_test, y_train, y_test = Data(source_file_name).splitTrainTest(test_size)

        #fix random seed for reproducibility
        np.random.seed(42)

        #disable warning
        tf.logging.set_verbosity(tf.logging.ERROR)
        
        #create model
        model = Sequential()
        model.add(LSTM(self.number_units))
        if self.number_dropout > 0:
            model.add(Dropout(self.number_dropout))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        #early stopping
        es_callback = EarlyStopping(monitor='val_loss', patience=3)
        
        #train model
        model.fit(X_train, y_train, validation_data=(X_test, y_test), callbacks = [es_callback], epochs=self.number_epochs, batch_size=self.number_batch_size, verbose=2, shuffle=False)

        return model
    
class Predict:

    def __init__(self, predict_file_name, predict_size):
        #read csv or excel data
        if predict_file_name[-3:] == 'csv':
            df = pd.read_csv(predict_file_name, parse_dates = ['timestamp'])
        else:
            df = pd.read_excel(predict_file_name, parse_dates = ['timestamp'])

        #mark stabil as 1
        df['stable'] = df['stable'].fillna(1)

        #set input and output
        X_stabil = df['count']
        y_stabil = df['stable']

        #set size of data to be predicted
        data_size = int(len(df.index) * float(predict_size))
        X_test = df['count'][-(data_size):]
        y_test = df['stable'][-(data_size):]

        #reshape input for model's requierment
        X_test = X_test.to_numpy()
        y_test = y_test.to_numpy()
        X_test = X_test.reshape((X_test.shape[0],1,1))
        self.y_test = y_test

        #get the trained model
        model = Model(number_units, number_epochs, number_batch_size, number_dropout).train()

        #predict
        result = model.predict_classes(X_test)
        self.result = result

    def result(self):
        #prepare the x_axis
        x_axis = np.arange(0, self.result.shape[0])

        #set the actual data above the predicted data with '+2' for visualize purpose
        expected_result = self.y_test + 2
        self.expected_result = expected_result

        #print classification report
        print(classification_report(self.y_test, self.result))
        
        #print confusion matrix
        self.confusionMatrix()
        
    def confusionMatrix(self):
        cm = confusion_matrix(self.y_test, self.result)
        print(cm)

if __name__ == '__main__':
    #source data(csv/excel), columns name should be = timestamp; count; stabil
    source_file_name = ''
    test_size = '0.3'

    #model
    number_units = 8 #add units to make more sensitive (LSTM)
    number_dropout = '0' #dropout to prevent overfitting / oversensitive
    number_epochs = 10
    number_batch_size = 20
    
    #predict and visualize
    predict_file_name = '' #predict data(csv/excel), columns name should be = timestamp; count; stabil
    predict_size = '0.3' #predict size(%) of the data from behind
    Predict(predict_file_name, predict_size).result()
