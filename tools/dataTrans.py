import os
import numpy as np
# audio
import librosa
import librosa.display
import sklearn
import torch 
from torch.utils.data import Dataset,DataLoader

def load_clip(filename):
    x, sr = librosa.load(filename, sr=22050)
    x = np.pad(x,(0,4 * sr - x.shape[0]),'constant')
    return x, sr

def extract_feature(filename):
    x, sr = load_clip(filename)
    mfccs = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=40)
    norm_mfccs = sklearn.preprocessing.scale(mfccs, axis=1)
    return norm_mfccs

def load_dataset(filenames):
    features, labels = np.empty((0,40,173)), np.empty(0)
    cnt = 0;
    cnt_all = len(filenames)
    
    for filename in filenames:
        mfccs = extract_feature(filename)
        features = np.append(features,mfccs[None],axis=0)
        cnt+=1
        if(cnt%100==0):
            print([str(cnt)+' / '+str(cnt_all)+' finished'])
        labels = np.append(labels, filename.split('\\')[1].split('-')[1])
    return np.array(features), np.array(labels, dtype=np.int)



class Mydataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.label = labels

    def __getitem__(self,index):
        return self.data[index],self.label[index]

    def __len__(self):
        return len(self.data)