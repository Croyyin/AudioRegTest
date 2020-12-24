import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.quantization import *

class CNN_1(nn.Module):
    def __init__(self):
        super(CNN_1, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=32,            
                kernel_size=3,              
                stride=1,                   
                
                padding=0,
            ),                              
            nn.ReLU(),                      
            
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(32, 32, 3, 1, 0),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )

        self.linear1 = nn.Linear(10496, 10)
        self.out = nn.Softmax(dim=1)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        x = x.view(x.size(0), -1)
        output = self.out(self.linear1(x))
        return output, x    

class CNN_2(nn.Module):
    def __init__(self):
        super(CNN_2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=0,)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0)  
        self.linear1 = nn.Linear(10496, 10)
        self.out = nn.Softmax(dim=1)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)

        x = x.view(x.size(0), -1)
        output = self.out(self.linear1(x))
        return output, x    

    def quantize(self, num_bits=8):
        self.qconv1 = QConv2d(self.conv1, qi=True, qo=True, num_bits=num_bits)
        self.qrelu1 = QReLU()
        self.qmaxpool2d_1 = QMaxPooling2d(kernel_size=2, stride=2, padding=0)
        self.qconv2 = QConv2d(self.conv2, qi=False, qo=True, num_bits=num_bits)
        self.qrelu2 = QReLU()
        self.qmaxpool2d_2 = QMaxPooling2d(kernel_size=2, stride=2, padding=0)
        self.qlinear1 = QLinear(self.linear1, qi=False, qo=True, num_bits=num_bits)

    # 前向计算，统计scale和zero_point
    def quantize_forward(self, x):
        x = self.qconv1(x)
        x = self.qrelu1(x)
        x = self.qmaxpool2d_1(x)
        x = self.qconv2(x)
        x = self.qrelu2(x)
        x = self.qmaxpool2d_2(x)

        x = x.view(x.size(0), -1)
        output = self.out(self.qlinear1(x))
        return output

    def freeze(self):
        self.qconv1.freeze()
        self.qrelu1.freeze(self.qconv1.qo)
        self.qmaxpool2d_1.freeze(self.qconv1.qo)
        self.qconv2.freeze(qi=self.qconv1.qo)
        self.qrelu2.freeze(self.qconv2.qo)
        self.qmaxpool2d_2.freeze(self.qconv2.qo)
        self.qlinear1.freeze(qi=self.qconv2.qo)

    def quantize_inference(self, x):
        qx = self.qconv1.qi.quantize_tensor(x)
        qx = self.qconv1.quantize_inference(qx)
        qx = self.qrelu1.quantize_inference(qx)
        qx = self.qmaxpool2d_1.quantize_inference(qx)
        qx = self.qconv2.quantize_inference(qx)
        qx = self.qrelu2.quantize_inference(qx)
        qx = self.qmaxpool2d_2.quantize_inference(qx)

        qx = qx.view(qx.size(0), -1)
        qx = self.qlinear1.quantize_inference(qx)
        qx = self.qlinear1.qo.dequantize_tensor(qx)
        output = self.out(qx)
        return output



class CNN_3(nn.Module):
    def __init__(self):
        super(CNN_3, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=3,              
                stride=1,                   
                
                padding=0,
            ),                              
            nn.ReLU(),                      
            
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        self.conv3 = nn.Sequential(         
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )

        self.linear1 = nn.Linear(3648, 300)
        self.linear2 = nn.Linear(300, 10)
        self.out = nn.Softmax(dim=1)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x.view(x.size(0), -1)
        output = self.out(self.linear2(self.linear1(x)))
        return output, x    


class CNN_4(nn.Module):
    def __init__(self):
        super(CNN_4, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=32,            
                kernel_size=5,              
                stride=1,                   
                
                padding=0,
            ),                              
            nn.ReLU(),                      
            
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(32, 32, 3, 1, 0),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )

        self.linear1 = nn.Linear(10496, 10)
        self.out = nn.Softmax(dim=1)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        x = x.view(x.size(0), -1)
        output = self.out(self.linear1(x))
        return output, x    