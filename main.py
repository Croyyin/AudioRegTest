import time
# basic handling
import os
import glob
import pickle
import numpy as np
# audio
import librosa
import librosa.display
import IPython.display
# normalization
import sklearn
import tools.dataTrans as dt

import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

from torch.utils.data import Dataset,DataLoader
from tools.dataTrans import Mydataset
from models import CNN

# 超参数
EPOCH = 200               
BATCH_SIZE = 64
LR = 0.001              
DOWNLOAD_MNIST = False
PATH = './models/trained/cnn_4/cnn_4_'

plt.style.use('ggplot')

parent_dir = './data/UrbanSound8K/audio/'
train_dir = 'trains/'
val_dir = 'val/'
test_dir = 'fold9/'
file_name = '*.wav'

train_files = glob.glob(os.path.join(parent_dir, train_dir, file_name))
val_files = glob.glob(os.path.join(parent_dir, val_dir, file_name))
test_files = glob.glob(os.path.join(parent_dir, test_dir, file_name))

# 数据转换
# train_x, train_y = dt.load_dataset(train_files)
# pickle.dump(train_x, open('./data/prc/trains_x.dat', 'wb'))
# pickle.dump(train_y, open('./data/prc/trains_y.dat', 'wb'))

# val_x, val_y = dt.load_dataset(val_files)
# pickle.dump(val_x, open('./data/prc/val_x.dat', 'wb'))
# pickle.dump(val_y, open('./data/prc/val_y.dat', 'wb'))

# test_x, test_y = dt.load_dataset(test_files)
# pickle.dump(test_x, open('./data/prc/test_x.dat', 'wb'))
# pickle.dump(test_y, open('./data/prc/test_y.dat', 'wb'))

# 数据加载
train_x = pickle.load(open('./data/prc/trains_x.dat', 'rb')).astype(np.float32)
train_y = pickle.load(open('./data/prc/trains_y.dat', 'rb')).astype(np.float32)
val_x = pickle.load(open('./data/prc/val_x.dat', 'rb')).astype(np.float32)
val_y = pickle.load(open('./data/prc/val_y.dat', 'rb')).astype(np.float32)
test_x = pickle.load(open('./data/prc/test_x.dat', 'rb')).astype(np.float32)
test_y = pickle.load(open('./data/prc/test_y.dat', 'rb')).astype(np.float32)


# 数据预处理
train_x = train_x.reshape((train_x.shape[0],1,train_x.shape[1],train_x.shape[2]))
val_x = val_x.reshape((val_x.shape[0],1,val_x.shape[1],val_x.shape[2]))
# test_x = test_x.reshape((test_x.shape[0],test_x.shape[1],test_x.shape[2],1))
val_x = torch.from_numpy(val_x)
val_y = torch.from_numpy(val_y)


mydataset = Mydataset(train_x, train_y)
dataloader = DataLoader(dataset=mydataset, shuffle=True, batch_size=BATCH_SIZE)

cnn = CNN.CNN_4()

# 优化器
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
# 损失函数
loss_func = nn.CrossEntropyLoss()

# 数据统计
train_data_size = train_x.shape[0]
val_data_size = val_x.shape[0]
train_loss = []
train_acc = []
val_loss = []
val_acc = []
epochs = [] 

time_start=time.time()
for epoch in range(EPOCH):
    
    
    # 初始化指标
    t_loss = 0
    v_loss = 0
    train_acc_count = 0
    val_acc_count = 0
    step_count = 0

    for step, (b_x, b_y) in enumerate(dataloader):

        output = cnn(b_x)[0]
        loss = loss_func(output, b_y.long())   
        optimizer.zero_grad()           
        loss.backward()                 
        optimizer.step()   

        t_loss += loss.data.numpy()
        
        train_pred_y = torch.max(output, 1)[1]
        train_acc_count += (train_pred_y == b_y).sum()
        step_count += 1

    # train loss
    t_loss = t_loss / step_count
    train_loss.append(t_loss)
    # val loss
    output1 = cnn(val_x)[0]               
    v_loss = loss_func(output1, val_y.long())  
    val_loss.append(v_loss.data.numpy())

    # train acc
    train_acc.append(float(train_acc_count) / float(train_data_size))

    # val acc
    val_pred_y = torch.max(output1, 1)[1]
    val_acc_count = (val_pred_y == val_y).sum()
    val_acc.append(float(val_acc_count) / float(val_data_size))
    
    epochs.append(epoch + 1)

    #保存
    torch.save(cnn.state_dict(), PATH + str(epoch + 1) + '.pt')
    # 打印进度
    print('epoch ' + str(epoch + 1) + ' already finished.')


time_end=time.time()
print('totally cost',time_end-time_start)

# loss 图
plt.subplot(1, 2, 1)
plt.title('Loss line chart')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(epochs, train_loss, 'b', label = 'Training loss')
plt.plot(epochs, val_loss, 'r', label = 'Validation loss')
plt.legend()

# acc 图
plt.subplot(1, 2, 2)
plt.title('Accuracy line chart')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.plot(epochs, train_acc, 'b', label = 'Training acc')
plt.plot(epochs, val_acc, 'r', label = 'Validation acc')
plt.legend()

plt.savefig('./result_cnn_4_trains_200.png', bbox_inches='tight')

plt.show()



# train_y = torch.zeros(train_y.shape[0], 10).scatter_(1, train_y, 1)
# val_y = torch.zeros(val_y.shape[0], 10).scatter_(1, val_y, 1)
# test_y = torch.zeros(test_y.shape[0], 10).scatter_(1, test_y, 1)

