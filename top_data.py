#!/usr/bin/env python
import os
import sys
import glob
import pandas
import numpy as np
from torch.utils.data import Dataset

def load_data(partition):
    nmax=50000
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'tops')
    all_data = []
    all_label = []
    filename='train.h5' if partition == 'train' else   'test.h5'
    #print "dir",DATA_DIR+'/'+filename
    store = pandas.HDFStore(DATA_DIR+'/'+filename)
    df = store.select("table")
    nmax=len(df)
    dflabel=df.filter(['is_signal_new'],axis=1)
    dfdata=df.drop(['truthE',  'truthPX',  'truthPY',  'truthPZ' ,'ttv' ,'is_signal_new'],axis=1)
    rdfdata=dfdata.values.reshape(nmax,200,4)
    store.close()
    all_data.append(rdfdata)
    all_label.append(dflabel.values)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
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
