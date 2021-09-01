import torch
import torch.nn as nn
import numpy.matlib as nm
from torch.optim import lr_scheduler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv(open('denoising_501_data.csv')).iloc[0:10000]

s0=np.array(df[['TT4']])
s = (s0 - np.min(s0))/(np.max(s0)-np.min(s0))
# print('原始数据：',list(s.reshape(1,len(s))[0]))
s = s.reshape((1,s.shape[0]))


'''绘制原始图'''
ori_data = np.array(s[0])
l = len(ori_data)
x_axis = np.arange(0, l, 1)
# 绘图
plt.plot(x_axis, ori_data)



Epoch = 50
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
        l_8 = int(self.input_num/8)
        l_16 = int(self.input_num/16)
        self.encoder = nn.Sequential(
            # 编码器
            nn.Linear(in_features=self.input_num,out_features=l_8),
            nn.LeakyReLU(),
            nn.Linear(in_features=l_8,out_features=l_16),
            nn.LeakyReLU(),
        )
        self.decoder = nn.Sequential(
            # 解码器
            nn.Linear(in_features=l_16, out_features=l_8),
            nn.LeakyReLU(),
            nn.Linear(in_features=l_8, out_features=self.input_num),
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
        loss1 = torch.square(torch.norm(x_pre-x_true,p=2))
        print("loss1的类型",type(loss1))
        loss2 = torch.norm(self.lamda*torch.matmul(self.d,torch.transpose(x_pre,0,1)),p=1)
        print("loss2的类型", type(loss2))
        # loss3 = torch.tensor(torch.norm(self.lamda*torch.matmul(self.d,torch.transpose(x_pre,0,1)),p=1))
        # print("loss3的类型", type(loss3))
        '''这里犯了一个错误，就是没有搞清楚torch.Tensor 和 torch.tensor，
            大写的首先明确一点，大写是python类，是torchFloatTensor的别名，是生成张量返回类但不能指定数据类型
            小写的仅仅是一个函数，是从其他类型data数据中做拷贝根据原始数据类型生成相应的tensor数据，他可以指定转换类型，也被用来生成新的张量
       
       PS：还有一个疑惑，3 个print输出都是同一类型，
       为什么加了一步torch.tensor的处理导致最后去噪后的数据无论lamda怎么改变，最后结果都一样，
        '''
        return loss1+loss2





autoencoder = AutoEncoder(Signal_Length)

optimizer = torch.optim.Adam(autoencoder.parameters(),lr=LR)
scheduler = lr_scheduler.LambdaLR(optimizer,lr_lambda = lambda i: 1/(i+1))
#loss_func = nn.MSELoss()
loss_func = myloss(lamda=0.01,d = D_tensor )

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
    '''
    retain_graph：
默认 retain_graph=False，
也就是反向传播之后这个计算图的内存会被释放，这样就没办法进行第二次反向传播了
每次 backward() 时，默认会把整个计算图free掉。一般情况下是每次迭代，只需一次 forward() 和一次 backward() 
前向运算forward() 和反向传播backward()是成对存在的，一般一次backward()也是够用的。
但是不排除，由于自定义loss等的复杂性，需要一次forward()，多个不同loss的backward()来累积同一个网络的grad,来更新参数。
于是，若在当前backward()后，不执行forward() 而可以执行另一个backward()，需要在当前backward()时，指定保留计算图，即backward(retain_graph)。

    '''
    optimizer.step()
    scheduler.step()
    # 查看数据
    if(epoch%5==0):
        # print('原始数据：', list(s.reshape(1, len(s))[0]))
        print("epoch",epoch)
        print('去噪数据',b_y)
        print('损失精度',loss)
        # 绘图



'''   绘制去噪后的数据图   '''

# 绘图pr
print( b_y[0].detach().numpy())
plt.plot(x_axis, b_y[0].detach().numpy())
plt.show()

