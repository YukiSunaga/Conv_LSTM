import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas
import sys

def j2r0(jfile, win_size=8, interval=1):
    win_size = int(win_size)
    with open(jfile, 'rb') as f:
        d = pickle.load(f)
    time = d['time']
    c = d['class']
    risk = []
    size = c[0].size
    o_size = int(np.floor((size - win_size) / interval) + 1)
    if o_size < 0:
        o_size = 1
    output_img = np.ones((4,o_size))
    o_time = []
    now_iter = 0
    print('\n')
    for s in range(o_size):
        t = s*interval
        for i in range(4):
            if t+win_size >= c[0].size:
                output_img[i,s] = np.mean(c[i][t:])
            else:
                output_img[i,s] = np.mean(c[i][t:t+win_size])
        if t+win_size >= c[0].size:
            o_time.append(time[-1])
        else:
            o_time.append(time[t+win_size-1])
        now_iter += 1
        sys.stdout.write("\r %d / %d" %(now_iter, o_size))
        sys.stdout.flush()
    risk = {}
    risk['time'] = np.array(o_time)
    for i in range(4):
        risk['pch'+str(i+1)] = output_img[i]

    return risk

def visualize_risk(risk, savefig=False, savepkl=False, filename='filename', loc='best',bbox_to_anchor=(1.01, 1)):
    time = pandas.to_datetime(risk['time'])
    fig, ax = plt.subplots(figsize=(13,5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:%M'))
    plt.xlabel('time')
    plt.ylabel('risk')
    plt.ylim(0.0, 1.0)
    ax.hlines(y=0.5, xmin=time[0], xmax=time[-1], linestyle='dashed')
    for i in range(4):
        plt.plot(time,risk['pch'+str(i+1)], label='ch'+str(i+1), lw=1.0, marker='.', ms=4.0)
    plt.legend(bbox_to_anchor=bbox_to_anchor, loc=loc)
    if savepkl:
        fn = filename
        if fn[-4:] == '.png':
            fn = fn[:-4]
        with open(fn + '.pkl', 'wb') as f:
            pickle.dump(risk, f, -1)
    if savefig:
        plt.tight_layout()
        plt.savefig(filename)
        plt.close('all')
    else:
        plt.tight_layout()
        plt.show()


def j2r0_4ch(jfile, win_size=8, interval=1):
    win_size = int(win_size)
    with open(jfile, 'rb') as f:
        d = pickle.load(f)
    time = d['time']
    c = d['class']
    risk = []
    size = c.size
    o_size = int(np.floor((size - win_size) / interval) + 1)
    if o_size < 0:
        o_size = 1
    output_img = np.ones((o_size))
    o_time = []
    now_iter = 0
    print('\n')
    for s in range(o_size):
        t = s*interval
        if t+win_size >= c.size:
            output_img[s] = np.mean(c[t:])
        else:
            output_img[s] = np.mean(c[t:t+win_size])
        if t+win_size >= c.size:
            o_time.append(time[-1])
        else:
            o_time.append(time[t+win_size-1])
        now_iter += 1
        sys.stdout.write("\r %d / %d" %(now_iter, o_size))
        sys.stdout.flush()
    risk = {}
    risk['time'] = np.array(o_time)
    risk['pch1-4'] = output_img

    return risk

def visualize_risk_4ch(risk, savefig=False, savepkl=False, filename='filename', loc='best',bbox_to_anchor=(1.01, 1)):
    time = pandas.to_datetime(risk['time'])
    fig, ax = plt.subplots(figsize=(13,5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:%M'))
    plt.xlabel('time')
    plt.ylabel('risk')
    plt.ylim(0.0, 1.0)
    ax.hlines(y=0.5, xmin=time[0], xmax=time[-1], linestyle='dashed')
    plt.plot(time,risk['pch1-4'], label='ch1-4', lw=1.0, marker='.', ms=4.0)
    plt.legend(bbox_to_anchor=bbox_to_anchor, loc=loc)
    if savepkl:
        fn = filename
        if fn[-4:] == '.png':
            fn = fn[:-4]
        with open(fn + '.pkl', 'wb') as f:
            pickle.dump(risk, f, -1)
    if savefig:
        plt.tight_layout()
        plt.savefig(filename)
        plt.close('all')
    else:
        plt.tight_layout()
        plt.show()
