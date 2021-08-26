

import pandas as pd
import threading
import torch.nn.modules
import torch.nn.functional as F
import numpy as np
from torch.autograd import  Variable
import machine_learning_fit_csv as fnt
import time
import inspect
import ctypes

testpoint_arr = ['滑油总管流量','滑油冷却器滑油进口温度','滑油冷却器1号滑油入口压力','滑油冷器冷却水进口流量','滑油冷却器1号冷却水入口温度','滑油冷却器1号冷却水出口温度','滑油总管温度']
short_t_arr = ['F1','TT1','PT1','F2','TT2','PT2','TT3']



class client_net(object):

    #神经网络类
    class Net(torch.nn.Module):
        def __init__(self, n_feature, n_hidden, n_hidden1,n_output):
            # 初始网络的内部结构
            super(client_net.Net,self).__init__()
            self.hidden = torch.nn.Linear(n_feature, n_hidden)
            self.hidden1 = torch.nn.Linear(n_hidden, n_hidden1)
            self.predict = torch.nn.Linear(n_hidden1, n_output)

        def forward(self, x):
            # 一次正向行走过程
            x = F.relu(self.hidden(x))
            x = F.relu(self.hidden1(x))
            x = self.predict(x)
            return x


    # 训练数据
    def train_net(self,x,y,iter_max,loss):
        x_data1 = torch.Tensor(x)
        y_data1 = torch.Tensor(y)
        # Variable是将tensor封装了下，用于自动求导使用
        x, y = Variable(x_data1), Variable(y_data1)
        global net
        net = self.Net(n_feature=6, n_hidden=5, n_hidden1=3, n_output=1)
        # print('------      启动训练      ------')
        loss_func = F.mse_loss
        global optimizer
        optimizer = torch.optim.SGD(net.parameters(), lr=0.0005,momentum=0.9)

        # 使用数据 进行正向训练，并对Variable变量进行反向梯度传播  启动10000次训练
        global cont
        cont = 0
        global state
        state = -1
        for t in range(iter_num):
            state = 0
            cont+=1
            # 使用全量数据 进行正向行走
            global prediction
            # prediction = net(x)
            prediction = net(x[t])
            # 填充nan
            prediction = torch.where(torch.isnan(prediction),torch.full_like(prediction,0),prediction)

            # if (torch.isnan(prediction)):
            #     prediction = torch.Tensor([0])
            #     prediction.requires_grad = True

            global loss0
            # loss0 = loss_func(prediction, y)
            loss0 = loss_func(prediction, y[t])
            optimizer.zero_grad()  # 清除上一梯度
            loss0.backward()  # 反向传播计算梯度
            optimizer.step()  # 应用梯度
        state = 1
        # torch.save(net.state_dict(),'model.pth')


        return net,state


    # 测试数据
    def test_net(self,x,y,net):
        x_data1 = torch.Tensor(x)
        y_data1 = torch.Tensor(y)
        # Variable是将tensor封装了下，用于自动求导使用
        x, y = Variable(x_data1), Variable(y_data1)
        # print('------      启动预测   ------')

        l = len(y)
        sum = 0
        for t in range(l):
            # 使用全量数据 进行正向行走
            global prediction
            prediction = net(x[t])
            if(torch.isnan(prediction)):
                prediction = torch.Tensor([0])
                prediction.requires_grad = True
            sum+=abs(float(prediction)-y[t])
        loss=sum/l
        print( {'state':2,'loss':loss,'ROC':{}})
        return {'state':2,'loss':loss,'ROC':{}}



#任务一：拼接三个CSV文件，仅保存特定测点，返回一个csv文件保存到aying里
def do_csv(file_name):
    '''
    filename:filename 就是outfile，add_csv生成的拼接文件
    #add_csv（）运行一次就不用再运行了

    '''
    # 看一下全部测点
    df = pd.read_csv(open(file_name))
    new_df = {}
    k = 0
    # 总共有多少个测点
    l = len(testpoint_arr)
    data = df[testpoint_arr].fillna(0)
    l1 = len(data[testpoint_arr[0]])
    ls = list(range(0, l1))
    for i in range(0, l):
        new_df[testpoint_arr[i]] = data[testpoint_arr[i]].loc[:]
        new_df[testpoint_arr[i]].index = ls
    # print(new_df)

    # 把汉字测点名称改为原来的字母
    # 字典型
    for i in range(0,l):
        new_df[short_t_arr[i]] = new_df[testpoint_arr[i]]
        del new_df[testpoint_arr[i]]
    # print(new_df)

    #生成一个do_csv.csv文件，保存在aying文件夹下
    pd.DataFrame(new_df).to_csv('do_csv.csv')


# 任务二 多线程
def main():
    print("开始")
    # 第一步，实例化
    cli = client_net()
    # 第二歩，csv数据处理
    # 2.1 读取拼接后csv文件
    df = pd.read_csv(open('final_cooler_csv.csv'))
    # 2.1 数据处理,提取出x和y 同时归一化处理
    x0 = df[['F1', 'TT1', 'PT1', 'F2', 'TT2', 'PT2']]
    y0 = np.array(df[['TT3']])
    print(y0)
    # 归一化处理x和y
    x1 = np.array(x0)
    global x_data
    x_data = minmax_normaliize(x1)
    y = (y0 - np.amin(y0)) / (np.amax(y0) - np.amin(y0))
    global y_data
    y_data = np.array(y)
    l1 = len(y_data)
    global l2
    l2 = int(2*l1/3)
    # print(x_data)
    # print(y_data)
    # 数据处理完毕


    while True:
        input0 = input()
        input1 = eval(input0)
        # print(input1)
        x = input1['type']
        # 第二歩 进入多线程
        if (x == 1):
            global iter_num
            global loss
            iter_num = input1['iter_max']
            loss = input1['loss']
        #开始训练数据
            t1 = threading.Thread(target=cli.train_net,args=(x_data[:l2],y_data[:l2],10000,0.1))
            # print("训练开始")
            t1.start()
            if(t1.is_alive()):
                print({'state':1,'train_num':l2})
            else:
                print({'state':-1})

        if (x == 2):
            def func2():
                # 正在训练模型
                if(state == 0):
                    # print('进行到第',cont,'次')
                    trainData = {'state':0,'trainData':{'F1':x_data[cont][0],'TT1':x_data[cont][1],'PT1':x_data[cont][2],'F2':x_data[cont][3],
                                'TT2':x_data[cont][4],'PT2':x_data[cont][5],'TT3':float(y0[cont])}, 'loss':loss0,'iter':cont}
                    print(trainData)

                elif(state == -1):
                    # print('未开始训练')
                    trainData = {'state':-1,'trainData':{'F1': 0, 'TT1': 0, 'PT1': 0, 'F2': 0, 'TT2': 0, 'PT2': 0,'TT3': 0},  'loss':0,'iter':0}
                    print(trainData)

                else:
                    # print('已结束训练')
                    trainData = {'state': 1,
                                 'trainData': {'F1':x_data[cont][0],'TT1':x_data[cont][1],'PT1':x_data[cont][2],'F2':x_data[cont][3],
                                'TT2':x_data[cont][4],'PT2':x_data[cont][5],'TT3':float(y0[cont])},
                                 'loss': loss0 ,'iter': iter_num}
                    print(trainData)

            # 输出当前训练状态 有格式要求
            t2 = threading.Thread(target=func2)
            t2.start()

        if (x == 3):
            # 暂停训练
            def func3():
                def _async_raise(tid, exctype):
                    """raises the exception, performs cleanup if needed"""
                    tid = ctypes.c_long(tid)
                    if not inspect.isclass(exctype):
                        exctype = type(exctype)
                    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
                    if res == 0:
                        raise ValueError("invalid thread id")
                    elif res != 1:
                        # """if it returns a number greater than one, you're in trouble,
                        # and you should call it again with exc=NULL to revert the effect"""
                        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
                        raise SystemError("PyThreadState_SetAsyncExc failed")

                def stop_thread(thread):
                    _async_raise(thread.ident, SystemExit)
                if(t1.is_alive()):
                    cont1 = cont
                    stop_thread(t1)
                    time.sleep(1)
                    cont2 = cont
                    if(cont1==cont2):
                        # print('成功暂停')
                        print({'state':1})
                    else:
                        print({'state':-1})
                else:
                    # print('t1未开始或已经结束')
                    print({'state':1})

            #     python 在3.3就已经废弃了线程的停止

            t3 = threading.Thread(target=func3)
            t3.start()

        if (x == 4):
            # 判断当前是否在训练，若训练完成则开始测试
            try:
                if(t1.is_alive()):
                    # print('训练还未结束')
                    print({'state':-1})

                else:
                    l3 = len(y_data[l2:-1])
                # 训练已经结束开始测试
                #     print('开始测试')
                    t3 = threading.Thread(target=cli.test_net,args=(x_data[l2:-1],y_data[l2:-1],net))
                    t3.start()
                    print({'state': 1,'test_num':l3})
            except UnboundLocalError:
                    # print('训练还未开始或训练异常中断，建议重新训练')
                    print({'state':-1})
        # 保存net
        if(x == 5):
            net.load_state_dict(torch.load('model.pth'))
            net.eval()
            # 打印模型的 state_dict
            print("Model's state_dict:")
            for param_tensor in net.state_dict():
                print(param_tensor, "\t", net.state_dict()[param_tensor].size())

            # 打印优化器的 state_dict
            print("Optimizer's state_dict:")
            for var_name in optimizer.state_dict():
                print(var_name, "\t", optimizer.state_dict()[var_name])



            pass

        # 动态更改模型结构
        if(x == 6):
            pass


# 归一化处理
def minmax_normaliize(data):
    l = data.shape[0]
    for  i in range(0,l):
        # print('data[i]',data[i])
        min  = np.amin(data[i])
        # print('mu',mu)
        max = np.amax(data[i])
        # print('std',std)
        data[i] = (data[i] - min)/(max-min)

    return data





if __name__ == '__main__':
    # {'type': 1, 'iter_max': 10, 'loss': 0.1}
    # {'type':2}
    # {'type':3}
    #{'type':4}
    # {'type':5}
    # do_csv('D:\workdemo1\现场数据\outfile.csv')
    main()