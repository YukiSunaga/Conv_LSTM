from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Conv1D, MaxPooling1D
from keras.models import load_model
from keras import backend as K



class Network():
    def __init__(self, x_train_shape, filters=16, kernel_size=5, pool_size=4, lstm_output_size=64):
        self.model = Sequential()
        #model.add(Embedding(max_features, embedding_size, input_length=maxlen))
        #model.add(Dropout(0.25))
        self.model.add(Conv1D(filters,
                         kernel_size,
                         input_shape=x_train_shape[1:],
                         padding='valid',
                         activation='relu',
                         strides=1))
        self.model.add(Conv1D(filters,
                         kernel_size,
                         padding='valid',
                         activation='relu',
                         strides=1))
        self.model.add(Conv1D(filters,
                         kernel_size,
                         padding='valid',
                         activation='relu',
                         strides=1))
        self.model.add(MaxPooling1D(pool_size=pool_size))
        self.model.add(LSTM(lstm_output_size))
        self.model.add(Dense(1))
        self.model.add(Activation('sigmoid'))

        self.model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        self.get_output = K.function([self.model.layers[0].input],
                                  [self.model.layers[-1].output])


    def train(self, x_train, y_train, x_test, y_test, batch_size=32, epochs=20):
        print('Train...')
        self.model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_test, y_test))
        score, acc = self.model.evaluate(x_test, y_test, batch_size=batch_size)
        print('Test score:', score)
        print('Test accuracy:', acc)
        return acc

    def save_params(self, model_name='convlstm_model', weights_name='convlstm_weights'):
        self.model.save(model_name + '.h5')
        self.model.save_weights(weights_name + '.h5')

    def predict(self, x, train_flg=False):
        output = self.get_output([x])[0]
        return output

    def classify(self, x, T=None, train_flg=False, one_or_zero=True):
        return self.predict(x)

    def load_params(self, model_name='convlstm_model', weights_name='convlstm_weights'):
        self.model = load_model(model_name)
        self.model.load_weights(weights_name)
