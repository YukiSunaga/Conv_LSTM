import numpy as np
import pickle
from convlstm import Network

import datetime
from evaluation.classify_whole_img import *
from evaluation.sens_spec import *

# Convolution
kernel_size = 5
filters = 16
pool_size = 4

# LSTM
lstm_output_size = 70

# Training
batch_size = 30
epochs = 10

evaluation=True

data_file = "../rvcnn_for_time_series/n_a_000-006_3_256_577_229[-1.0 1.0].pkl"
with open(data_file, 'rb') as f:
    d = pickle.load(f)

(x_train, y_train), (x_test, y_test) = (d['train_img'], d['train_label']), (d['test_img'], d['test_label'])
x_train = x_train.reshape((x_train.shape[0], -1, 1))
x_test = x_test.reshape((x_test.shape[0], -1, 1))

'''
x_train = x_train[:100]
y_train = y_train[:100]
x_test = x_test[:100]
y_test = y_test[:100]
'''

x_train_shape = x_train.shape
win_ch, win_col = x_train.shape[2], x_train.shape[1]

print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

network = Network(x_train_shape, filters=filters, kernel_size=kernel_size, pool_size=pool_size, lstm_output_size=lstm_output_size)
acc = network.train(x_train, y_train, x_test, y_test, batch_size=batch_size, epochs=epochs)

today = datetime.datetime.today()
file_name = "log_" + today.strftime("%Y%m%d%H%M")
path = "result/" + today.strftime("%Y%m%d%H%M") + "/"
os.makedirs(path)

network.save_params(model_name=path+'convlstm_model', weights_name=path+'convlstm_weights')

with open(path + "h_params" + file_name + ".txt", "a") as f:
    f.write('program : train_mulfreq_conv_net.py \n')
    f.write("data : " + data_file +'\n')
    f.write("x_shape = " + str(x_train_shape) +'\n')
    f.write("final test acc : " + str(acc))


if evaluation:
    mag = 2
    r_mag = 1
    if x_train_shape[2] == 1:
        calc_spse(network, path=path, win_size=win_col, stride=int(win_col/ mag), r_win_size=int(r_mag*mag*16*30*60/(4*win_col)), one_or_zero=False)
    elif x_train_shape[2] == 4:
        calc_spse_4ch(network, path=path, win_size=win_col, stride=int(win_col/ mag), r_win_size=int(r_mag*mag*16*30*60/(4*win_col)))
