#!/usr/bin/env python
import os
import sys
import glob
import pandas
import numpy as np
from torch.utils.data import Dataset

def load_data(partition):
    nmax=40000
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'tops')
    all_data = []
    all_label = []
    filename='train.h5' if partition == 'train' else   'test.h5'
    #print "dir",DATA_DIR+'/'+filename
    store = pandas.HDFStore(DATA_DIR+'/'+filename)
    df = store.select("table",stop=nmax)
    nmax=len(df)
    drop=[]
    for i in range(0,200):
        df['PHI_' +str(i)]=np.arctan2(df['PY_'+str(i)],df['PX_'+str(i)])
        df['ETA_' +str(i)]=0.5*np.log((df['E_'+str(i)]+df['PZ_'+str(i)])/(df['E_'+str(i)]-df['PZ_'+str(i)]))
        df['PT_'  +str(i)]=np.sqrt(df['PX_'+str(i)]*df['PX_'+str(i)]+df['PY_'+str(i)]*df['PY_'+str(i)])
        df['P_'   +str(i)]=np.sqrt(df['PX_'+str(i)]*df['PX_'+str(i)]+df['PY_'+str(i)]*df['PY_'+str(i)]+df['PZ_'+str(i)]*df['PZ_'+str(i)])
        df['M_'   +str(i)]=np.sqrt(df['E_'+str(i)]*df['E_'+str(i)]-df['PX_'+str(i)]*df['PX_'+str(i)]-df['PY_'+str(i)]*df['PY_'+str(i)]-df['PZ_'+str(i)]*df['PZ_'+str(i)])
        drop+=['PX_'+str(i),'PY_'+str(i),'PZ_'+str(i),'E_'+str(i),"P_"+str(i)]
    df=df.fillna(0)
    dflabel=df.filter(['is_signal_new'],axis=1)
    dfdata=df.drop(['truthE',  'truthPX',  'truthPY',  'truthPZ' ,'ttv' ,'is_signal_new']+drop,axis=1)
    dfdata=dfdata.values.reshape(nmax,200,4)
    all_data.append(dfdata.astype('float32'))
    all_label.append(dflabel.values)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    store.close()
    return all_data, all_label


class TopData(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.label = load_data(partition)
        self.num_points = num_points

    def __getitem__(self, item):
        return self.data[item][:self.num_points], self.label[item]

    def __len__(self):
        return self.data.shape[0]

if __name__ == '__main__':
    train = TopData(200)
    test  = TopData(200, 'test')
    for data, label in train:
        print(data.shape)
        print(label.shape)
