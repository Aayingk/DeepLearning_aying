
import pandas as pd
import torch.nn.modules
import torch.nn.functional as F
import numpy as np
from torch.autograd import  Variable
import json


device = 'cuda' if torch.cuda.is_available() else 'cpu'
# 归一化处理
def minmax_normaliize(data):
    l = data.shape[0]
    for  i in range(0,l):
        # print('data[i]',data[i])
        min  = np.amin(data[i])
        # print('mu',mu)
        max = np.amax(data[i])
        # print('std',std)
        if(max==min):
            data[i] = 0
        else:
            data[i] = (data[i] - min)/(max-min)
    return data



# 根据输入动态定义神经网络的结构
class Net(torch.nn.Module):
    m = 0
    n = 0
    p = 0
    def __init__(self,net_struct, save_filaname, ACC_LOSS, MAX_ITER):
        super(Net, self).__init__()
        self.net_dir = save_filaname
        self.ACC_LOSS = ACC_LOSS
        self.MAX_ITER = MAX_ITER
        self.model = torch.nn.Sequential()
        self.net_struct = net_struct
        self.len0 = len(net_struct.keys())
    '''  对应某特定层的添加语句，在确实所有可添加的层后再完善'''
    def Linear(self, var1, var2):
        self.model.add_module('Linear' + str(self.m), torch.nn.Linear(var1, var2))
        self.m += 1
    def LogSigmoid(self):
        self.model.add_module('LogSigmoid' + str(self.n), torch.nn.LogSigmoid())
        self.n += 1
    def ReLU(self):
        self.model.add_module('ReLU' + str(self.p), torch.nn.ReLU())
        self.p += 1
    '''   创建网络   '''
    def excu(self):
        for i in range(self.len0):
            if (self.net_struct[str(i + 1)][0] == 'Linear'):
                eval('self.Linear(self.net_struct[str(i+1)][1],self.net_struct[str(i+1)][2])')
            else:
                eval('self.' + self.net_struct[str(i + 1)][0] + '()')
        print(self.model)
        return self.model
    ''' 网络前向传播'''
    def forward(self, x):
        # 一次正向行走过程
        '''
        x = F.relu(self.hidden0(x))
        x = F.relu(self.hidden(x))
        x = F.relu(self.hidden1(x))
        x = self.predict(x)
        return x
        '''
        predict = self.model.forward(x)
        return predict



a =1
'''一个可动态更改模型结构（包括层数，层类型，单层节点数，激活函数类型）实体类的网络'''
class client_net(object):
    def __init__(self,my_struct,is_new):
        # 注意，该处初始化最后整合为一个json变量，将上述变量存储在json中
        self.my_struct = my_struct
        self.is_new = is_new
        if(self.is_new == "true"):
            self.net = Net(my_struct,input[ "save_filename"]
                         ,input["ACC_LOSS"],input["MAX_ITER"]).to(device)
            self.net.excu()
        else:
            self.net = torch.load(input["save_filename"])
            print(self.net)


    def train_net(self,x, y):
        x_data1 = torch.Tensor(x)
        y_data1 = torch.Tensor(y)
        x, y = Variable(x_data1), Variable(y_data1)
        loss_func = F.mse_loss
        optimizer = torch.optim.SGD(self.net.parameters(), lr=0.0005, momentum=0.99)
        # 动态调整学习率
        scheduler_1 = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0005, max_lr=0.005)
        # 使用数据 进行正向训练，并对Variable变量进行反向梯度传播  启动iter_max次训练
        leny = len(y)
        for i in range(self.net.MAX_ITER):
            predict_yy0 = []
            sum_loss = 0
            for t in range(0, leny):
                # 使用全量数据 进行正向行走
                prediction = self.net(x[t])
                predict_yy0.append(prediction.item())
                loss0 = loss_func(prediction, y[t])
                optimizer.zero_grad()  # 清除上一梯度
                loss0.backward(retain_graph=True)  # 反向传播计算梯度
                optimizer.step()  # 应用梯度
                sum_loss += loss0.item()
            avg_loss = sum_loss / leny
            scheduler_1.step()
            # print("第%d次迭代的学习率： %f" % (i, optimizer.param_groups[0]['lr']))
            if (avg_loss <= self.net.ACC_LOSS):
                # 保存网络
                torch.save(self.net, self.net.net_dir)
                break
        # 保存整个网络
        torch.save(self.net, self.net.net_dir)
        print(self.net)
        print('训练结束')


    # 测试数据
    # 注意测试数据需要把当成参数net传进来
    def test_net(self,x,y):
        x_data1 = torch.Tensor(x)
        y_data1 = torch.Tensor(y)
        # Variable是将tensor封装了下，用于自动求导使用
        x, y = Variable(x_data1), Variable(y_data1)
        l = len(y)
        sum = 0
        pred_data = []
        for t in range(l):
            # 使用全量数据 进行正向行走
            prediction = self.net(x[t])
            pred_data.append(prediction)
            if(torch.isnan(prediction)):
                prediction = torch.Tensor([0])
                prediction.requires_grad = True
            sum += abs(float(prediction)-y[t])
        loss = sum/l
        print("预测数据",pred_data)
        print("平均误差",loss)
        print("预测结束")





def main(json_input):
    input = json.loads(json_input)


    '''处理数据，根据传进来的数据地址获取数据，此处默认数据是以csv文件保存的'''
    def deal_with_data(file_dir):
        # 数据准备
        df = pd.read_csv(open(file_dir)).iloc[0:1001]
        # 2.1 数据处理,提取出x和y 同时归一化处理
        x0 = df[['F1', 'TT1', 'PT1', 'F2', 'TT2', 'TT3']]
        y0 = np.array(df[['TT4']])
        # 归一化处理x和y
        x1 = np.array(x0)
        x_data = minmax_normaliize(x1)
        y_data = (y0 - np.amin(y0)) / (np.amax(y0) - np.amin(y0))
        return x_data, y_data
    x_data, y_data = deal_with_data(input["train_data_filename"])


    '''用户端来的自定义网络机构new_struct格式如下：
            new_struct = {"layer_num":4(从图去理解此处应该填几)
                          "layer_name":['Linear','ReLU','Linear','ReLU','Linear','ReLU'],
                          "point_num":[6,5,3,1]（此处注意首尾两个数字保证和输入输出的维数相同）
    '''
    def deal_new_struct(new_struct):
        key_0 = list(np.arange(1, len(new_struct["layer_name"])))
        keys = []
        for i in key_0:
            keys.append(str(i))
        values = [[] for i in range(len(new_struct["layer_name"]))]
        my_newstruct = dict(zip(keys, values))
        #  {'1': [], '2': [], '3': [], '4': [], '5': []}
        j = 0
        for i in keys:
            my_newstruct[i].append(new_struct["layer_name"][j])
            j+=1
        m = 0
        n = 1
        for i in range(len(new_struct["layer_name"]) - 1):
            if (i % 2 == 0):
                my_newstruct[keys[i]].append(new_struct["point_num"][m])
                my_newstruct[keys[i]].append(new_struct["point_num"][m + 1])
                m += 1
                continue
            else:
                my_newstruct[keys[i]].append(new_struct["point_num"][n])
                my_newstruct[keys[i]].append(new_struct["point_num"][n])
                n += 1
                continue
        return my_newstruct
    my_struct0 = deal_new_struct(input["net_struct"])
    # my_struct={1:['Linear',len(x_data[0]),5],2:['ReLU',5,5],3:['Linear',5,3],4:['ReLU',3,3],5:['Linear',3,len(y_data[0])]}


    '''实例化主类和神经网络类'''
    cli = client_net(my_struct0,is_new=input["is_new"])
    # 训练模型
    cli.train_net(x_data,y_data)
    # 预测
    cli.test_net(x_data,y_data)





# 传输的数据参照格式

input = {"type":"SEND_DATA",
         "train_data_filename":"./denoising_501_data.csv",
         "save_filename":"./FNN_net.pkl",
         "is_new":"true",
         "net_struct":{"layer_num": 4,
                        "layer_name": ['Linear', 'ReLU', 'Linear', 'ReLU', 'Linear', 'ReLU'],
                        "point_num": [6, 5, 3, 1]
                      },
         "ACC_LOSS":0.001,
         "MAX_ITER":100
         }
json_input = json.dumps(input)



if __name__ == '__main__':
    main(json_input)