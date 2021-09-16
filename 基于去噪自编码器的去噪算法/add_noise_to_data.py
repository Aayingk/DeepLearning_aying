import torch
import torch.nn as nn
import numpy.matlib as nm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

x = np.linspace(0,2 * np.pi, 1000, endpoint=True)
y = np.array(np.sin(x))
# 给数据加噪声
mu  = 0
sigma = 0.12
for i in range(x.size):
    x[i]+=random.gauss(mu,sigma)
    y[i]+=random.gauss(mu,sigma)


s0 = np.array(y)
s = (s0 - np.min(s0))/(np.max(s0)-np.min(s0))
s = s.reshape((1,s.shape[0]))
'''绘制原始图'''
ori_data = np.array(s[0])
l = len(ori_data)
x_axis = np.arange(0, l, 1)


Epoch = 20
LR = 0.01

"张量维度要调整为（1，n）"

signal=torch.tensor(s,requires_grad=True).to(torch.float32)
Signal_Length=signal[0].shape[0]

"""
D为全变差矩阵，构造D
"""
D=nm.identity(Signal_Length,dtype=int)
D=D*(-1)
for i in range(Signal_Length-1):
    D[i,i+1]=1
#将numpy中的矩阵数据类型转换为tensorflow中的张量类型
D_tensor=torch.tensor(D).to(torch.float32)


class AutoEncoder(nn.Module):
    def __init__(self,len_data):
        super(AutoEncoder,self).__init__()
        self.input_num = len_data
        self.encoder = nn.Sequential(
            # 编码器
            nn.Linear(in_features=self.input_num,out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128,out_features=64),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            # 解码器
            nn.Linear(in_features=64, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=self.input_num),
            nn.Sigmoid(),
        )


    def forward(self,x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class myloss(torch.nn.Module):
    def __init__(self,lamda,d):
        super().__init__()
        self.lamda = lamda
        self.d = d

    def forward(self,x_pre,x_true):
        loss1 = torch.square(torch.norm(x_pre-x_true,p=1.0))
        loss2 = torch.norm(self.lamda*torch.matmul(self.d,torch.transpose(x_pre,0,1)),p=1)

        return loss1+loss2







autoencoder = AutoEncoder(Signal_Length)

optimizer = torch.optim.Adam(autoencoder.parameters(),lr=LR)
#scheduler = lr_scheduler.LambdaLR(optimizer,lr_lambda = lambda i: 1/(i+1))
#loss_func = nn.MSELoss()
loss_func = myloss(lamda=20,d = D_tensor)

b_x = signal[0]
b_x = b_x.contiguous()
# 开始训练
for epoch in range(Epoch):
    # 自编码器去噪体现在数据x和y都是x的数据
    b_y = autoencoder(b_x.view(1,Signal_Length))
    loss = loss_func(b_y,b_x)
    # 常规操作
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    #scheduler.step()
    # 查看数据
    if(epoch%5==0):
        # print('原始数据：', list(s.reshape(1, len(s))[0]))
        print('epoch',epoch)
        print('去噪数据',b_y)
        print('损失精度',loss)
        # 绘图
        #plt.scatter(x_axis, b_y[0].detach().numpy())
        #plt.show()


'''   绘制去噪后的数据图   '''

# 绘图
plt.plot(x_axis, ori_data)
plt.plot(x_axis, b_y[0].detach().numpy())

plt.show()

