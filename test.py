import os
import pickle
import numpy as np

import torch
import torchvision
import matplotlib.pyplot as plt

from torch.utils.data import Dataset,DataLoader
from tools.dataTrans import Mydataset
from models import CNN
from tools.function import direct_quantize

BATCH_SIZE = 64
LOOP = 16

train_x = pickle.load(open('./data/prc/trains_x.dat', 'rb')).astype(np.float32)
train_y = pickle.load(open('./data/prc/trains_y.dat', 'rb')).astype(np.float32)
test_x = pickle.load(open('./data/prc/test_x.dat', 'rb')).astype(np.float32)
test_y = pickle.load(open('./data/prc/test_y.dat', 'rb')).astype(np.float32)



train_x = train_x.reshape((train_x.shape[0],1,train_x.shape[1],train_x.shape[2]))
test_x = test_x.reshape((test_x.shape[0],1,test_x.shape[1],test_x.shape[2]))
test_x = torch.from_numpy(test_x)
test_y = torch.from_numpy(test_y)

test_dataset = Mydataset(test_x, test_y)
test_loader = DataLoader(dataset=test_dataset, shuffle=True, batch_size=BATCH_SIZE)

train_dataset = Mydataset(train_x, train_y)
train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=BATCH_SIZE)

cnn = CNN.CNN_2()
cnn.load_state_dict(torch.load('models/trained/cnn_2/cnn_2_192.pt'))
cnn_list = []

correct = 0
for i, (data, target) in enumerate(test_loader, 1):
    cnn.eval()
    output = cnn(data)[0]
    pred = output.argmax(dim=1, keepdim=True)
    correct += pred.eq(target.view_as(pred)).sum().item()

full_acc = correct / len(test_loader.dataset)
fulls_acc = []
for i in range(LOOP):
    fulls_acc.append(full_acc)


qbits_acc = []
num_bits_sign = []    

for num_bits in range(1,17):

    _cnn = CNN.CNN_2()
    _cnn.load_state_dict(torch.load('models/trained/cnn_2/cnn_2_192.pt'))
    _cnn.eval()

    num_bits_sign.append(num_bits)
    
    _cnn.quantize(num_bits=num_bits)
    direct_quantize(_cnn, train_loader)
    _cnn.freeze()

    correct = 0
    q_acc = 0
    for i, (data, target) in enumerate(test_loader, 1):
        output = _cnn.quantize_inference(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    q_acc = correct / len(test_loader.dataset)
    print(q_acc)
    qbits_acc.append(q_acc)

    print(str(num_bits) + ' bits quantization finish')


# num_bits = 7
# cnn = cnn_list[0]
# cnn.eval()
# num_bits_sign.append(num_bits)

# cnn.quantize(num_bits=num_bits)
# direct_quantize(cnn, train_loader)
# cnn.freeze()
# correct = 0
# q_acc = 0
# for i, (data, target) in enumerate(test_loader, 1):
#     output = cnn.quantize_inference(data)
#     pred = output.argmax(dim=1, keepdim=True)
#     correct += pred.eq(target.view_as(pred)).sum().item()
# q_acc = correct / len(test_loader.dataset)
# qbits_acc.append(q_acc)
# print(str(num_bits) + ' bits quantization finish')

# print(fulls_acc)
# print(qbits_acc)

plt.title('Q_Accuracy')
plt.xlabel('bits')
plt.ylabel('accuracy')
plt.plot(num_bits_sign, fulls_acc, 'b', label = 'Full accuracy')
plt.plot(num_bits_sign, qbits_acc, 'r', label = 'Quantization accuracy')
plt.legend()


plt.savefig('./result_cnn_2_bits_full.png', bbox_inches='tight')

plt.show()