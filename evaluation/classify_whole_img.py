# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import scipy.misc
import sys
import pickle
import pandas

colormap = plt.cm.rainbow
bmax = 32767
bmin = -32768

def pd(filename, nmax, nmin):
    f = open(filename, 'r')
    d = pandas.read_csv(f)
    dn = {}
    for i in range(4):
        dn['pch'+str(i+1)] = d['pch'+str(i+1)].values
        dn['pch'+str(i+1)] = (dn['pch'+str(i+1)] - bmin) / (bmax - bmin) * (nmax - nmin) + nmin

    return [d['timepx'], dn['pch1'], dn['pch2'], dn['pch3'], dn['pch4']]


def pd4ch(filename, nmax, nmin):
    f = open(filename, 'r')
    d = pandas.read_csv(f)
    for i in range(4):
        d['pch'+str(i+1)] = (d['pch'+str(i+1)] - bmin) / (bmax - bmin) * (nmax - nmin) + nmin

    return [d['timepx'], np.array([d['pch1'], d['pch2'], d['pch3'], d['pch4']])]


def classify_image_ch_p(network, image, output_size, win_row, win_col, stride=3, psize=32):
    #under constraction
    if image.ndim == 2:
        image = image.reshape(1, image.shape[0], image.shape[1])
    channel, row, col = image.shape
    out_row = int(numpy.floor((row - win_row) / stride) + 1)
    out_col = int(numpy.floor((col - win_col) / stride) + 1)
    osize = out_row * out_col
    piter = int(osize / psize)
    rem = osize - psize*piter

    output_img = numpy.empty((out_row*out_col, ), dtype='int')

    now_iter = 0

    T = network.teacher_label()

    oomuro_list = numpy.zeros((1,2), dtype=int)
    first_flg = True

    for i in range(piter):
        input_wins = numpy.zeros((psize, channel, win_row, win_col), dtype='complex64')
        for p in range(psize):
            curwin = i*psize + p
            x = int(curwin/out_col)
            y = int(curwin - x*out_col)
            x*=stride
            y*=stride
            #print('%d, %d' %(x,y))
            input_wins[p] = image[:,x:x+win_row, y:y+win_col]
        #print(input_wins.shape)
        if GPU:
            input_wins = to_gpu(input_wins)
        wins_class = network.classify(input_wins, T)
        output_img[i*psize:(i+1)*psize] = wins_class
        now_iter += 1
        sys.stdout.write("\r %d / %d" %(now_iter, piter+1))
        #time.sleep(1)
        #sys.stdout.write("\r %d,  %d, %d,  %d, %d,  %d, %d" %(wins_class[wins_class==0].size,wins_class[wins_class==1].size,wins_class[wins_class==2].size,wins_class[wins_class==3].size,wins_class[wins_class==4].size,wins_class[wins_class==5].size,wins_class[wins_class==6].size))
        sys.stdout.flush()

    input_wins = numpy.zeros((rem, channel, win_row, win_col), dtype='complex64')
    for p in range(rem):
        curwin = piter*psize + p
        x = int(curwin/out_col)
        y = int(curwin - x*out_col)
        x*=stride
        y*=stride
        #print('%d, %d' %(x,y))
        input_wins[p] = image[:,x:x+win_row, y:y+win_col]
    if GPU:
        input_wins = to_gpu(input_wins)
    wins_class = network.classify(input_wins, T)
    output_img[piter*psize:] = wins_class
    now_iter += 1
    sys.stdout.write("\r %d / %d" %(now_iter, piter+1))

    sys.stdout.flush()

    output_img = output_img.reshape((out_row, out_col))

    return output_img, oomuro_list


def classify_wave2(network, wave, win_size=128, stride=3, psize=32, one_or_zero=True):
    size = wave.size
    osize = int(np.floor((size - win_size) / stride) + 1)
    piter = int(osize / psize)
    rem = osize - psize*piter
    if one_or_zero:
        output_img = -1*np.ones(osize, dtype='int')
    else:
        output_img = -1*np.ones(osize)
    now_iter = 0

    for i in range(piter):
        input_wins = np.zeros((psize, win_size, 1))
        for p in range(psize):
            y = i*psize + p
            y*=stride
            y = int(y)
            #print('%d, %d' %(x,y))
            input_wins[p] = wave[y:y+win_size].reshape((win_size, 1))
        if one_or_zero:
            wins_class = network.classify(input_wins).flatten()
        else:
            wins_class = network.classify(input_wins, one_or_zero=False).flatten()
        output_img[i*psize:(i+1)*psize] = wins_class
        now_iter += 1
        sys.stdout.write("\r %d / %d" %(now_iter, piter+1))
        #time.sleep(1)
        #sys.stdout.write("\r %d,  %d, %d,  %d, %d,  %d, %d" %(wins_class[wins_class==0].size,wins_class[wins_class==1].size,wins_class[wins_class==2].size,wins_class[wins_class==3].size,wins_class[wins_class==4].size,wins_class[wins_class==5].size,wins_class[wins_class==6].size))
        sys.stdout.flush()

    if not rem == 0:
        input_wins = np.zeros((rem, win_size, 1))
        for p in range(rem):
            y = piter*psize + p
            y*=stride
            y = int(y)
            #print('%d, %d' %(x,y))
            input_wins[p] = wave[y:y+win_size].reshape((win_size, 1))
        if one_or_zero:
            wins_class = network.classify(input_wins).flatten()
        else:
            wins_class = network.classify(input_wins, one_or_zero=False).flatten()
        output_img[piter*psize:] = wins_class
        now_iter += 1
        sys.stdout.write("\r %d / %d" %(now_iter, piter+1))

        sys.stdout.flush()

    return output_img

def classify_csv(network, file, filepath, patientnum='01-001',win_size=128, stride=3, nmax=1., nmin=-1., one_or_zero=True):
    d = pd(file, nmax, nmin)
    size = d[0].size
    o_size = int(np.floor((size - win_size) / stride) + 1)
    time = []
    for s in range(o_size):
        t = s*stride
        time.append(d[0][t])
    time = np.array(time)
    c = []
    for i in range(4):
        print('\nch'+str(i+1))
        c.append(classify_wave2(network, d[i+1], win_size=win_size, stride=stride, one_or_zero=one_or_zero))
    savet = {}
    savet['time'] = time
    savet['class'] = c
    with open(filepath + 'class_' + patientnum + '.pkl', 'wb') as f:
        pickle.dump(savet, f, -1)
    fig, ax = plt.subplots(4,1)
    time = pandas.to_datetime(time)
    for i in range(4):
        ax[i].xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:%M'))
        ax[i].scatter(time, c[i], s=0.1, color='b')
    plt.savefig(filepath + 'class_' + patientnum + '.png')
    #plt.pause(1e-5)
    plt.close('all')


def classify_wave2_4ch(network, wave, win_size=128, stride=3, psize=32):
    size = wave.shape[1]
    osize = int(np.floor((size - win_size) / stride) + 1)
    piter = int(osize / psize)
    rem = osize - psize*piter
    output_img = -1*np.ones(osize, dtype='int')
    now_iter = 0

    for i in range(piter):
        input_wins = np.zeros((psize, 4, 1, win_size))
        for p in range(psize):
            y = i*psize + p
            y*=stride
            y = int(y)
            #print('%d, %d' %(x,y))
            input_wins[p] = wave[:,y:y+win_size].reshape((4,1,win_size))
        wins_class = network.classify(input_wins).flatten()
        output_img[i*psize:(i+1)*psize] = wins_class
        now_iter += 1
        sys.stdout.write("\r %d / %d" %(now_iter, piter+1))
        #time.sleep(1)
        #sys.stdout.write("\r %d,  %d, %d,  %d, %d,  %d, %d" %(wins_class[wins_class==0].size,wins_class[wins_class==1].size,wins_class[wins_class==2].size,wins_class[wins_class==3].size,wins_class[wins_class==4].size,wins_class[wins_class==5].size,wins_class[wins_class==6].size))
        sys.stdout.flush()

    if not rem == 0:
        input_wins = np.zeros((rem, 4, 1, win_size))
        for p in range(rem):
            y = piter*psize + p
            y*=stride
            y = int(y)
            #print('%d, %d' %(x,y))
            input_wins[p] = wave[:,y:y+win_size].reshape((4,1,win_size))
        wins_class = network.classify(input_wins).flatten()
        output_img[piter*psize:] = wins_class
        now_iter += 1
        sys.stdout.write("\r %d / %d" %(now_iter, piter+1))

        sys.stdout.flush()

    return output_img


def classify_csv_4ch(network, file, filepath, patientnum='01-001',win_size=128, stride=3, nmax=1., nmin=-1.):
    d = pd4ch(file, nmax, nmin)
    size = d[0].size
    o_size = int(np.floor((size - win_size) / stride) + 1)
    time = []
    for s in range(o_size):
        t = s*stride
        time.append(d[0][t])
    time = np.array(time)
    c = classify_wave2_4ch(network, d[1], win_size=win_size, stride=stride)
    savet = {}
    savet['time'] = time
    savet['class'] = c
    with open(filepath + 'class_' + patientnum + '.pkl', 'wb') as f:
        pickle.dump(savet, f, -1)
    fig, ax = plt.subplots(1,1)
    time = pandas.to_datetime(time)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:%M'))
    ax.scatter(time, c, s=0.1, color='b')
    plt.savefig(filepath + 'class_' + patientnum + '.png')
    #plt.pause(1e-5)
    plt.close('all')



def classify_image(network, image, output_size, win_row=28, win_col=28, stride=3):
    row, col = image.shape
    out_row = int(np.floor((row - win_row) / stride) + 1)
    out_col = int(np.floor((col - win_col) / stride) + 1)

    output_img = np.empty((out_row, out_col), dtype='int')

    max_iter = int(out_row*out_col)
    now_iter = 0

    T = network.teacher_label()

    for r in range(out_row):
        for c in range(out_col):
            x = r*stride
            y = c*stride
            input_win = image[x:x+win_row, y:y+win_col].reshape(1,1,win_row,win_col)
            output_img[r,c] = network.classify(input_win, T)
            now_iter += 1
            sys.stdout.write("\r %d / %d" %(now_iter, max_iter))
            sys.stdout.flush()

    return output_img

def classify_image_ch(network, image, output_size, win_row, win_col, stride=3):
    if image.ndim == 2:
        image = image.reshape(1, image.shape[0], image.shape[1])
    channel, row, col = image.shape
    out_row = int(np.floor((row - win_row) / stride) + 1)
    out_col = int(np.floor((col - win_col) / stride) + 1)

    output_img = np.empty((out_row, out_col), dtype='int')

    max_iter = int(out_row*out_col)
    now_iter = 0

    T = network.teacher_label()

    oomuro_list = np.zeros((1,2), dtype=int)
    first_flg = True

    for r in range(out_row):
        for c in range(out_col):
            x = r*stride
            y = c*stride
            input_win = image[:,x:x+win_row, y:y+win_col].reshape(1,channel,win_row,win_col)
            win_class = network.classify(input_win, T)
            output_img[r,c] = win_class
            if win_class == 6:
                if first_flg:
                    oomuro_list[0] = [x, y]
                    first_flg = False
                else:
                    oomuro_list = np.concatenate((oomuro_list, ([[x,y]])), axis=0)
            now_iter += 1
            sys.stdout.write("\r %d / %d" %(now_iter, max_iter))
            sys.stdout.flush()

    return output_img, oomuro_list

def classify(network, img, filepath, win_row=28, win_col=28):
    print('classifying...')
    #image = np.load('FujiHakone_mini.npy')
    '''
    data_file = "sea_other_dataset_comp"
    with open(data_file + ".pkl", 'rb') as f:
            dataset = pickle.load(f)

    (x_train, t_train), (x_test, t_test) = (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label'])
    x = x_train[100:200].transpose(2,0,1,3).reshape((28, -1))
    t = t_train[100:200]
    print(t)
    print(t.shape)
    t = to_one_hot_label(t, network.out_size)
    print(network.loss(x_train[100:200], t))
    '''
    image = np.load(img)
    output_img = classify_image(network, image, network.out_size)


    #scipy.misc.imsave(filepath + 'output.png', output_img)
    np.save(filepath + 'output.npy', output_img)
    plt.imsave(filepath + 'output.png', output_img, cmap=colormap)





def classify_ch(network, img, filepath, win_row=28, win_col=28, stride=3):
    print('classifying...')

    image = np.load(img)
    output_img, oomuro_list = classify_image_ch(network, image, network.out_size, win_row=win_row, win_col=win_col, stride=stride)
    np.save(filepath + 'output.npy', output_img)
    plt.imsave(filepath + 'output.png', output_img, cmap=colormap)
    np.save(filepath + 'oomuro_list.npy', oomuro_list)


def classify_wave_ch(network, image, output_size, win_row, win_col, stride=3):
    if image.ndim == 1:
        image = image.reshape(1, 1, image.shape[0])
    channel, row, col = image.shape
    out_row = int(np.floor((row - win_row) / stride) + 1)
    out_col = int(np.floor((col - win_col) / stride) + 1)

    output_img = np.empty((out_row, out_col), dtype='int')

    max_iter = int(out_row*out_col)
    now_iter = 0

    T = network.teacher_label()
    first_flg = True

    for r in range(out_row):
        for c in range(out_col):
            x = r*stride
            y = c*stride
            input_win = image[:,x:x+win_row, y:y+win_col].reshape(1,channel,win_row,win_col)
            win_class = network.classify(input_win, T)
            output_img[r,c] = win_class
            now_iter += 1
            sys.stdout.write("\r %d / %d" %(now_iter, max_iter))
            sys.stdout.flush()
    sys.stdout.write('\n')

    return output_img


def classify_wave(network, img, filepath, win_row=1, win_col=128, stride=128):
    print('classifying...')

    wave = np.load(img)
    output_img = classify_wave_ch(network, wave, network.out_size, win_row=win_row, win_col=win_col, stride=stride)
    np.save(filepath + 'output.npy', output_img)
    plt.imsave(filepath + 'output.png', output_img, cmap=colormap)


if __name__ == '__main__':
    network = Network()
    network.load_params("params.pkl")
    classify(network)
