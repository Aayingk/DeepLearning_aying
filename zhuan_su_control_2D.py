import pandas as pd
import json
import numpy as np
import torch.nn.modules
import torch.nn.functional as F
from torch.autograd import  Variable
from sklearn.cluster  import KMeans
import random

def minmax_normaliize(data):
    l = data.shape[0]
    for i in range(0, l):
        # print('data[i]',data[i])
        min = np.amin(data[i])
        # print('mu',mu)
        max = np.amax(data[i])
        # print('std',std)
        if (max == min):
            data[i] = 0
        else:
            data[i] = (data[i] - min) / (max - min)
    return data


'''           第一步先分类，对同类数据做相关性分析           '''
def return_clustering_data(filename):
    # 准备数据
    df = pd.read_csv(filename,encoding="gbk").drop("Time",axis=1)
    temp_list = []
    for columns_name ,columns_data in df.iteritems():
        temp_list.append(list(columns_data))
    # data_dict 是字典型的用于测试数据聚类算法的数据
    data_dict = dict(zip(df.keys(),temp_list))
    clustering_train_data = {'type':'SEND_DATA',
                       'data':json.dumps(data_dict)}

    class Clustering:
        def __init__(self, clustering_train_data):
            self.type = clustering_train_data['type']
            # clustering_train_data['data']是json格式，为了方便实用，在这里直接转换为字典型
            self.data = json.loads(clustering_train_data['data'])

        def deal_with_data(self):
            '''    阶段一：数据准备阶段       '''
            '''          处理数据           '''
            # 将json格式的数据转换为dataframe.再进行归一化处理
            temp_df = pd.DataFrame(self.data)
            # print(temp_df)
            final_data = np.array(temp_df)

            return final_data

        '''          训练聚类模型          '''
        '''从node端获取簇数 k 和 模型保存地址'''

        def train_model(self, k, dir_model):
            '''
                   k：分类种类数
                   x：train_data
                   '''
            train_data = self.deal_with_data()
            k = int(k)
            self.clf = KMeans(n_clusters=k)
            self.clf.fit(train_data)
            centers = self.clf.cluster_centers_  # 数据的中心点
            labels = self.clf.labels_  # 每个数据所述分组
            # print('数据中心点\n', centers)
            # print('数据标签：\n', labels)
            data = []
            for i in range(0, k):
                data.append([])
            for j in range(0, len(labels)):
                k = labels[j]
                data[k].append(train_data[j])
            # print('分类数据：\n', data)
            # 将分类后的数据每一类转成一个CSV文件
            for i in range(0, k):
                df = pd.DataFrame(data[k],columns=list(self.data.keys()))
                # 将分类数据保存在csv文件中
                df.to_csv('clu_data'+str(k)+'.csv',index = False)
            return data
    clustering = Clustering(clustering_train_data)
    k = 4
    # 训练模型并保存
    dir_model = "./clf.pkl"
    clustering.train_model(k, dir_model)


'''                第二歩相关性分析                        '''
def corr(filename,testpoint):
    data = pd.read_csv(filename)
    matrix = data.corr(method='spearman').fillna(axis=0,value=0)
    # 对角线赋值为1
    list1 = list(matrix.keys())
    for i in range(0, int(matrix.__len__())):
        matrix[list1[i]].iloc[i] = 1
    tp_corr = matrix[[testpoint]]
    corr_tp_set = []
    for index,row in tp_corr.iterrows():
        if(abs(row.values[0])>=0.5):
            corr_tp_set.append(index)
    corr_tp_set.remove(testpoint)
    print(corr_tp_set)
    return corr_tp_set

'''                第三步，使用当前测点集合进行预测泵转速                   '''
def predict_zhuansu(filename,testpoints,corr_tp_set):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 第一步处理数据
    # 3.1 读取拼接后csv文件
    df0 = pd.read_csv(open(filename))
    # 打乱处理
    ll = len(df0.index)
    pre_list = list(np.arange(0, ll))
    pre_index = random.sample(pre_list, ll)
    df = df0.iloc[pre_index]
    # 3.2 数据处理,提取出x和y 同时归一化处理
    data = np.array(df[corr_tp_set+testpoints])
    x0 = np.array(df[corr_tp_set])
    y0 = np.array(df[testpoints])
    # 归一化处理x和y
    norm_data = minmax_normaliize(data)
    x_data = norm_data[:,0:len(corr_tp_set)]
    y_data = norm_data[:,0:len(testpoints)]


    # 第二歩，定义神经网络
    class client_net(object):
        #神经网络类
        class Net(torch.nn.Module):
            def __init__(self, n_feature, n_hidden, n_hidden1,n_output):
                # 初始网络的内部结构
                super(client_net.Net,self).__init__()
                self.linear_relu_stack = torch.nn.Sequential(
                    torch.nn.LayerNorm(n_feature),
                    torch.nn.Linear(n_feature,n_hidden),
                    torch.nn.ReLU(),
                    torch.nn.Linear(n_hidden, n_hidden1),
                    torch.nn.ReLU(),
                    torch.nn.Linear(n_hidden1, n_output),
                )
            def forward(self, x):
                # 一次正向行走过程
                '''
                x = F.relu(self.hidden0(x))
                x = F.relu(self.hidden(x))
                x = F.relu(self.hidden1(x))
                x = self.predict(x)
                return x
                '''
                predict = self.linear_relu_stack(x)
                return predict

        def __init__(self):
            global net
            net = self.Net(n_feature=len(corr_tp_set), n_hidden=100, n_hidden1=150, n_output=2).to(device)


        # 训练数据
        def train_net(self,x,y,iter_max,acc):
            x_data1 = torch.Tensor(x)
            y_data1 = torch.Tensor(y)
            y000 = torch.Tensor(y[:,0])
            y00 = torch.reshape(y000,(len(y000),1))
            y111 = torch.Tensor(y[:,1])
            y11 = torch.reshape(y111,(len(y111),1))
            # Variable是将tensor封装了下，用于自动求导使用
            x, y, y0, y1 = Variable(x_data1), Variable(y_data1),Variable(y00),Variable(y11)

            # print('------      启动训练      ------')
            loss_func = F.mse_loss
            # 优化器选择
            optimizer = torch.optim.Adam(net.parameters(),lr = 0.0005)
            # 动态调整学习率
            '''
                  stepsize是学习率下降的间隔数，gamma下降倍数
            '''
            scheduler_1 = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,step_size=5,gamma=0.5,last_epoch=-1)
            # 使用据 进行正向训练，并对Variable变量进行反向梯度传播  启动iter_max次训练
            leny = len(y)
            for i in range(iter_max):
                sum_loss = 0
                for t in range(0,leny):
                    # 使用全量数据 进行正向行走
                    loss0 = loss_func(net(x[t]), y[t])
                    optimizer.zero_grad()  # 清除上一梯度
                    loss0.backward()  # 反向传播计算梯度
                    optimizer.step()  # 应用梯度
                    sum_loss += loss0.item()
                avg_loss = sum_loss/leny
                scheduler_1.step()
                if(avg_loss<=acc):
                    torch.save(net.state_dict(),"predict_feed.pth")
                    print("训练结束1")
                    return net
                    break
                torch.save(net.state_dict(), "predict_feed.pth")
                print("训练结束2")
                return net


        # 测试数据
        def test_net(self,x,y,net_dir,acc):
            net = self.Net(n_feature=len(corr_tp_set), n_hidden=100, n_hidden1=150, n_output=2).to(device)
            net.load_state_dict(torch.load(net_dir))
            x_data1 = torch.Tensor(x)
            y_data1 = torch.Tensor(y)
            # Variable是将tensor封装了下，用于自动求导使用
            x, y = Variable(x_data1), Variable(y_data1)
            l = len(y)
            sum = 0
            loss_func = F.mse_loss
            predict_yy = []
            sum_loss= 0
            for t in range(l):
                # 使用全量数据 进行正向行走
                prediction = net(x[t])
                predict_yy.append(prediction)
                loss0 = loss_func(prediction, y[t])
                sum_loss+=loss0
                if(loss0.item()<acc):
                    sum+=1
                    # acc0 满足精度要求的数据占比
                    acc0 = sum/l
            print('满足精度要求的数据比例为：\n',acc0)
            print("平均损失为：\n",sum_loss/l)
            # print("真实数据为：\n",y)
            print("预测数据为：\n",predict_yy)


    # 第三步，实例化并调用训练预测函数
    cli = client_net()
    cli.train_net(x_data[:20000],y_data[0:20000],10000,0.001)
    cli.test_net(x_data[20000:-1],y_data[20000:-1],"predict_feed.pth",0.001)




if __name__ == '__main__':
    # 先执行第一条语句，保存分类后的数据,执行一次后不必再执行
    # return_clustering_data("scan_data2.csv")
    # 返回相关性测点集合
    corr_tp_set = corr("clu_data2.csv","电动滑油泵2号变频泵转速指令")
    predict_zhuansu("scan_data2.csv",["电动滑油泵1号变频泵转速指令","电动滑油泵2号变频泵转速指令"],corr_tp_set)



