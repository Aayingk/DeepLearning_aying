import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv(open('denoising_501_data.csv')).iloc[0:10000]
x_0 = np.array(df['TT4'])
# 归一化处理
x_1 = (x_0 - np.min(x_0))/(np.max(x_0)-np.min(x_0))
#x_1 = x_0
x= torch.Tensor(x_1)
print('原始数据',x)
x_axis = np.arange(0, len(x), 1)
y_axis = np.array(x_1)
# 绘图
plt.plot(x_axis, y_axis)
plt.show()

Epoch = 100
LR = 0.01


class AutoEncoder(nn.Module):
    def __init__(self,len_data):
        super(AutoEncoder,self).__init__()
        self.input_num = len_data
        l_8 = int(self.input_num/8)
        l_16 = int(self.input_num/16)
        self.encoder = nn.Sequential(
            # 编码器
            nn.Linear(in_features=self.input_num,out_features=l_8),
            nn.Sigmoid(),
            nn.Linear(in_features=l_8,out_features=l_16),
            nn.Sigmoid(),
        )
        self.decoder = nn.Sequential(
            # 解码器
            nn.Linear(in_features=l_16, out_features=l_8),
            nn.Sigmoid(),
            nn.Linear(in_features=l_8, out_features=self.input_num),
            nn.Sigmoid(),
        )


    def forward(self,x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class myloss(nn.Module):
    def __init__(self,lamda):
        super().__init__()
        self.lamda = lamda
    def forward(self,x_pre,x_true):
        Dx = 0
        for i in range(len(x_pre)-1):
            Dx += torch.abs(x_pre[i+1]-x_pre[i]) 
        Dx = self.lamda * Dx
        distance = torch.pow(torch.sum(torch.pow(x_pre - x_true,2)),0.5)
        loss = Dx + distance
        return loss





autoencoder = AutoEncoder(len(x))
optimizer = torch.optim.Adam(autoencoder.parameters(),lr=LR)
scheduler = lr_scheduler.LambdaLR(optimizer,lr_lambda = lambda i: 1/(i+1))
#loss_func = nn.MSELoss()
loss_func = myloss(lamda=1)

b_y = torch.Tensor(x)
# 开始训练
for epoch in range(Epoch):
    # 自编码器去噪体现在数据x和y都是x的数据
    b_x = autoencoder(b_y)
    loss = loss_func(b_x,b_y)
    # 常规操作
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    # 查看数据
    if(epoch%5==0):
        print('去噪数据',b_x)
        print('精度损失',loss)



x_axis_2 = np.arange(0, len(x), 1)
y_axis_2 = b_x.detach().numpy()
# 绘图
plt.plot(x_axis_2, y_axis_2)
plt.show()
