import numpy as np
import pandas
import datetime
import os
from evaluation.classify_whole_img import *
from evaluation.judge2risk import *


partici_lis0 = ["01-006(TTM14)", "01-006b(TTM17)"]
partici_lis1 = ["01-0" + format(i, '02d') for i in range(7,18)]
partici_lis2 = ["02-001", "02-002"]
partici_lis3 = ["03-00" + str(i) for i in range(1,5)]
partici_lis6 = ["06-001", "06-002", "06-003"]
partici_lis5 = ["05-001"]

partici_lis_tr = ["01-00" + str(i) for i in range(1,6)] + partici_lis0

partici_lis_ev = partici_lis1 + partici_lis2 + partici_lis3 + partici_lis6 + partici_lis5



'''
partici_lis0 = ["01-006(TTM14)", "01-006b(TTM17)"]
partici_lis1 = ["01-0" + format(i, '02d') for i in range(15,16)]

partici_lis_ev = partici_lis0 + partici_lis1

partici_lis_tr = ["01-00" + str(i) for i in range(1,3)]
'''

def read_ref_file(ref_filepath="evaluation\\解析用データセット(まとめ)_20181002.xlsx", sensor_flg=False):
    d = pandas.read_excel(ref_filepath, sheet_name=1,  usecols=[0,1,6,20, 21, 22,23,31]).dropna(subset=["時間(EDC)"])
    d['datetime'] = (d['日付'].astype(str)).str.cat((d['時間(EDC)'].astype(str)), sep=' ')

    id = np.array(d["症例ID"])
    cls = np.array(d["色調異常あり"])
    cls[cls!=1] = 0
    dt = np.array(d["datetime"], dtype='datetime64[s]')

    if sensor_flg:
        sf1 = np.array(d['センサー1\nフラグ*'])
        sf2 = np.array(d['センサー2\nフラグ*'])
        sf3 = np.array(d['センサー3\nフラグ*'])
        sf4 = np.array(d['センサー4\nフラグ*'])

    if sensor_flg:
        return id, dt, cls, sf1, sf2, sf3, sf4
    return id, dt, cls

def read_risk_file(result_file="result/201902141452/risk_01-001.pkl"):
    with open(result_file, 'rb') as f:
        risk = pickle.load(f)
    t = risk['time']
    dt = []
    for i in range(t.size):
        dt.append(datetime.datetime.strptime(t[i], "%Y/%m/%d %H:%M:%S"))
    dt = np.array(dt, dtype="datetime64[s]")
    risk['time'] = dt
    return risk

def calc_risk(network, path, raw_pul_path="D:\\YukiSunaga\\臨床データ\\raw\\" ,win_col=256,  stride=256, r_win_size=64, overwrite=False, one_or_zero=True):
    for p in partici_lis_tr:
        print('\n\n ' + p)
        if (not os.path.exists(path + 'class_'+ p + '.pkl')) or overwrite:
            classify_csv(network=network, file= raw_pul_path + p + "_pul_raw.csv", filepath=path, patientnum=p, win_size=win_col, stride=stride, one_or_zero=one_or_zero)
        if (not os.path.exists(path + 'risk_' + p + '.pkl')) or overwrite:
            visualize_risk(j2r0(path + 'class_'+ p + '.pkl', win_size=r_win_size), savefig=True, savepkl=True, filename=path + 'risk_' + p + '.png')
    for p in partici_lis_ev:
        print('\n\n ' + p)
        if (not os.path.exists(path + 'class_'+ p + '.pkl')) or overwrite:
            classify_csv(network=network, file= raw_pul_path + p + "_pul_raw.csv", filepath=path, patientnum=p, win_size=win_col, stride=stride, one_or_zero=one_or_zero)
        if (not os.path.exists(path + 'risk_' + p + '.pkl')) or overwrite:
            visualize_risk(j2r0(path + 'class_'+ p + '.pkl', win_size=r_win_size), savefig=True, savepkl=True, filename=path + 'risk_' + p + '.png')


def calc_risk_4ch(network, path, raw_pul_path="D:\\YukiSunaga\\臨床データ\\raw\\",win_col=256,  stride=256, r_win_size=64, overwrite=False):
    for p in partici_lis_tr:
        print('\n\n ' + p)
        if (not os.path.exists(path + 'class_'+ p + '.pkl')) or overwrite:
            classify_csv_4ch(network=network, file= raw_pul_path + p + "_pul_raw.csv", filepath=path, patientnum=p, win_size=win_col, stride=stride)
        if (not os.path.exists(path + 'risk_' + p + '.pkl')) or overwrite:
            visualize_risk_4ch(j2r0_4ch(path + 'class_'+ p + '.pkl', win_size=r_win_size), savefig=True, savepkl=True, filename=path + 'risk_' + p + '.png')
    for p in partici_lis_ev:
        print('\n\n ' + p)
        if (not os.path.exists(path + 'class_'+ p + '.pkl')) or overwrite:
            classify_csv_4ch(network=network, file= raw_pul_path + p + "_pul_raw.csv", filepath=path, patientnum=p, win_size=win_col, stride=stride)
        if (not os.path.exists(path + 'risk_' + p + '.pkl')) or overwrite:
            visualize_risk_4ch(j2r0_4ch(path + 'class_'+ p + '.pkl', win_size=r_win_size), savefig=True, savepkl=True, filename=path + 'risk_' + p + '.png')


def find_closest_dt(dtlist, risk, find_range=1800):
    evt_list = []
    t = risk['time']
    if dtlist.size == 1:
        tpres = t - dtlist[0]
        am = np.argmin(np.abs(tpres))
        evt_list.append(am)
    else:
        for i, dt in enumerate(dtlist):
            if i == 0:
                tpres = t - dt
                tnex = t - dtlist[i+1]
                am = np.argmin(np.abs(tpres))
                if np.abs(tpres[am]) > np.abs(tnex[am]) or np.abs(tpres[am]) > datetime.timedelta(0,find_range):
                    evt_list.append(-1)
                else:
                    evt_list.append(am)
            elif i == dtlist.size - 1:
                tpres = t - dt
                tprev = t - dtlist[i-1]
                am = np.argmin(np.abs(tpres))
                if np.abs(tpres[am]) > np.abs(tprev[am]) or np.abs(tpres[am]) > datetime.timedelta(0,find_range):
                    evt_list.append(-1)
                else:
                    evt_list.append(am)
            else:
                tpres = t - dt
                tprev = t - dtlist[i-1]
                tnex = t - dtlist[i+1]
                am = np.argmin(np.abs(tpres))
                if np.abs(tpres[am]) > np.abs(tprev[am]) or np.abs(tpres[am]) > np.abs(tnex[am]) or np.abs(tpres[am]) > datetime.timedelta(0,find_range):
                    evt_list.append(-1)
                else:
                    evt_list.append(am)

    return evt_list


def sesp_to_excel(idlist, dtlist, clslist, risk_dic, risk_tgt_time, res_risk_list, path):
    flg = True
    for id in partici_lis_ev:
        if flg:
            time = risk_dic[id]['time']
            rtt = np.array(time[risk_tgt_time[id]])
            risk = np.array(res_risk_list[id])
            idl = np.array(idlist[idlist==id])
            dtl = np.array(dtlist[idlist==id])
            cll = np.array(clslist[idlist==id])
            flg = False
        else:
            time = risk_dic[id]['time']
            rtt = np.concatenate((rtt, np.array(time[risk_tgt_time[id]])))
            risk = np.concatenate((risk,np.array(res_risk_list[id])))
            idl = np.concatenate((idl,np.array(idlist[idlist==id])))
            dtl = np.concatenate((dtl,np.array(dtlist[idlist==id])))
            cll = np.concatenate((cll,np.array(clslist[idlist==id])))


    alarm = np.zeros(risk.shape, dtype=int)
    alarm[risk==-1] = -1
    alarm[risk>=0.5] = 1
    df = pandas.DataFrame()
    df['id'] = idl
    df['EDC time'] = dtl
    df['risk target time'] = rtt
    df['diagnosis'] = cll
    df['risk'] = risk
    df['alarm'] = alarm

    with pandas.ExcelWriter(path + 'risk.xlsx') as writer:
        df.to_excel(writer)



def calc_spse(network, path, ref="evaluation\\解析用データセット(まとめ)_20181002.xlsx",win_size=256, stride=256, r_win_size=64, calcrisk=True, one_or_zero=True, rt=0.5, risk_mode='max'):
    if calcrisk:
        calc_risk(network, path, win_col=win_size, stride=stride, r_win_size=r_win_size, one_or_zero=one_or_zero)

    partici_lis = partici_lis_ev

    risk_dic = {}
    for id in partici_lis:
        risk_dic[id] = read_risk_file(path + 'risk_' + id + '.pkl')


    idlist, dtlist, clslist, sf1, sf2, sf3, sf4 = read_ref_file(ref, sensor_flg=True)
    risk_tgt_time = {}
    for id in partici_lis:
        risk_tgt_time[id] = find_closest_dt(dtlist[idlist==id], risk_dic[id])

    res_risk_list = {}
    for id in idlist:
        if id in partici_lis:
            rl = []
            for i, tgt in enumerate(risk_tgt_time[id]):
                if tgt == -1:
                    rl.append(-1)
                else:
                    r1,r2,r3,r4 = risk_dic[id]['pch1'][tgt], risk_dic[id]['pch2'][tgt],risk_dic[id]['pch3'][tgt],risk_dic[id]['pch4'][tgt]
                    rm = []
                    if sf1[idlist==id][i] == 1:
                        r1 = -1
                    else:
                        rm.append(r1)
                    if sf2[idlist==id][i] == 1:
                        r2 = -1
                    else:
                        rm.append(r2)
                    if sf3[idlist==id][i] == 1:
                        r3 = -1
                    else:
                        rm.append(r3)
                    if sf4[idlist==id][i] == 1:
                        r4 = -1
                    else:
                        rm.append(r4)
                    if r1 == -1 and r2 == -1 and r3 == -1 and r4 == -1:
                        rl.append(-1)
                    elif risk_mode == 'max':
                        rl.append(np.max(np.array([r1,r2,r3,r4])))
                    else:
                        rm = np.array(rm)
                        rl.append(np.mean(rm))
            res_risk_list[id] = rl
        else:
            rl = [-1 for i in range(idlist[idlist==id].size)]
            res_risk_list[id] = rl

    confmat = np.zeros((2,2), dtype=int)
    for id in partici_lis:
        clsid = clslist[idlist==id]
        rrl = res_risk_list[id]
        for i, cls in enumerate(clsid):
            if rrl[i] == -1:
                pass
            elif rrl[i] >= rt and cls == 1:
                confmat[0,0] += 1
            elif rrl[i] >= rt and cls == 0:
                confmat[0,1] += 1
            elif rrl[i] < rt and cls == 1:
                confmat[1,0] += 1
            elif rrl[i] < rt and cls == 0:
                confmat[1,1] += 1

    sensitivity = confmat[0,0] / (confmat[0,0] + confmat[1,0])
    specificity = confmat[1,1] / (confmat[1,1] + confmat[0,1])
    J = sensitivity + specificity - 1

    print('\n')
    print(confmat)
    print("sensitivity : %f, specificity : %f" %(sensitivity, specificity))
    print("J : %f" %(J))

    with open(path + "sensitivity_specificity.txt", "a") as f:
        f.write(str(confmat))
        f.write('\nsensitivity : ' + str(sensitivity) + '\n')
        f.write('specificity : ' + str(specificity) + '\n')
        f.write('J : ' + str(J))


    sesp_to_excel(idlist, dtlist, clslist, risk_dic, risk_tgt_time, res_risk_list, path)

    return confmat, sensitivity, specificity



def calc_spse_4ch(network, path, ref="evaluation\\解析用データセット(まとめ)_20181002.xlsx",win_size=256, stride=256, r_win_size=64):
    calc_risk_4ch(network, path, win_col=win_size, stride=stride, r_win_size=r_win_size)

    partici_lis = partici_lis_ev

    risk_dic = {}
    for id in partici_lis:
        risk_dic[id] = read_risk_file(path + 'risk_' + id + '.pkl')


    idlist, dtlist, clslist = read_ref_file(ref)
    risk_tgt_time = {}
    for id in partici_lis:
        risk_tgt_time[id] = find_closest_dt(dtlist[idlist==id], risk_dic[id])

    res_risk_list = {}
    for id in idlist:
        if id in partici_lis:
            rl = []
            for i, tgt in enumerate(risk_tgt_time[id]):
                if tgt == -1:
                    rl.append(-1)
                else:
                    rl.append(risk_dic[id]['pch1-4'][tgt])
            res_risk_list[id] = rl
        else:
            rl = [-1 for i in range(idlist[idlist==id].size)]
            res_risk_list[id] = rl

    confmat = np.zeros((2,2), dtype=int)
    for id in partici_lis:
        clsid = clslist[idlist==id]
        rrl = res_risk_list[id]
        for i, cls in enumerate(clsid):
            if rrl[i] == -1:
                pass
            elif rrl[i] >= 0.5 and cls == 1:
                confmat[0,0] += 1
            elif rrl[i] >= 0.5 and cls == 0:
                confmat[0,1] += 1
            elif rrl[i] < 0.5 and cls == 1:
                confmat[1,0] += 1
            elif rrl[i] < 0.5 and cls == 0:
                confmat[1,1] += 1

    sensitivity = confmat[0,0] / (confmat[0,0] + confmat[1,0])
    specificity = confmat[1,1] / (confmat[1,1] + confmat[0,1])
    J = sensitivity + specificity - 1

    print('\n')
    print(confmat)
    print("sensitivity : %f, specificity : %f" %(sensitivity, specificity))
    print("J : %f" %(J))

    with open(path + "sensitivity_specificity.txt", "w") as f:
        f.write(str(confmat))
        f.write('\nsensitivity : ' + str(sensitivity) + '\n')
        f.write('specificity : ' + str(specificity) + '\n')
        f.write('J : ' + str(J))

    sesp_to_excel(idlist, dtlist, clslist, risk_dic, risk_tgt_time, res_risk_list, path)

    return confmat, sensitivity, specificity



def calc_spse_nr(path, ref="evaluation\\解析用データセット(まとめ)_20181002.xlsx"):
    partici_lis = partici_lis_ev

    risk_dic = {}
    for id in partici_lis:
        risk_dic[id] = read_risk_file(path + 'risk_' + id + '.pkl')


    idlist, dtlist, clslist = read_ref_file(ref)
    risk_tgt_time = {}
    for id in partici_lis:
        risk_tgt_time[id] = find_closest_dt(dtlist[idlist==id], risk_dic[id])

    res_risk_list = {}
    for id in idlist:
        if id in partici_lis:
            rl = []
            for i, tgt in enumerate(risk_tgt_time[id]):
                if tgt == -1:
                    rl.append(-1)
                else:
                    r1,r2,r3,r4 = risk_dic[id]['pch1'][tgt], risk_dic[id]['pch2'][tgt],risk_dic[id]['pch3'][tgt],risk_dic[id]['pch4'][tgt]
                    rl.append(np.max(np.array([r1,r2,r3,r4])))
            res_risk_list[id] = rl
        else:
            rl = [-1 for i in range(idlist[idlist==id].size)]
            res_risk_list[id] = rl

    confmat = np.zeros((2,2), dtype=int)
    for id in partici_lis:
        clsid = clslist[idlist==id]
        rrl = res_risk_list[id]
        for i, cls in enumerate(clsid):
            if rrl[i] == -1:
                pass
            elif rrl[i] >= 0.5 and cls == 1:
                confmat[0,0] += 1
            elif rrl[i] >= 0.5 and cls == 0:
                confmat[0,1] += 1
            elif rrl[i] < 0.5 and cls == 1:
                confmat[1,0] += 1
            elif rrl[i] < 0.5 and cls == 0:
                confmat[1,1] += 1

    sensitivity = confmat[0,0] / (confmat[0,0] + confmat[0,1])
    specificity = confmat[1,1] / (confmat[1,1] + confmat[1,0])

    print('\n')
    print(confmat)
    print("sencitivity : %f, specificity : %f" %(sensitivity, specificity))


    return confmat, sensitivity, specificity, idlist, dtlist, clslist, risk_tgt_time, res_risk_list, risk_dic
