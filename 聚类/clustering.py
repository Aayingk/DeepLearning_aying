
from sklearn import metrics
import pickle
from sklearn.metrics.pairwise import pairwise_distances_argmin
import pandas as pd
from sklearn.cluster  import KMeans
import json
import scipy.spatial.distance
import numpy
import numpy as np
import scipy.cluster.vq
import matplotlib.pyplot as plt
import time
EuclDist = scipy.spatial.distance.euclidean

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

'''  在正式交付版本中没有这部分，这部分的作用在于给出用户给出的数据参考格式  '''

df = pd.read_csv(open('denoising_501_data.csv')).iloc[0:1001]
data_1 = df[['F1' ,'TT1' ,'PT1' ,'F2' ,'TT2' ,'TT3','TT4']]
# print(data_1)
temp_list = []
for columns_name ,columns_data in data_1.iteritems():
    temp_list.append(list(columns_data))
# print(temp_list)
# data_dict 是字典型的用于测试数据聚类算法的数据
data_dict = dict(zip(['F1' ,'TT1' ,'PT1' ,'F2' ,'TT2' ,'TT3','TT4'],temp_list))
# print(data_dict)

clustering_train_data = {'type':'SEND_DATA',
                   'data':json.dumps(data_dict)}
print(clustering_train_data)

pre_data = [[1, 0.48699788, 0, 0.7100488, 0.11844756, 0.17191174, 0.25296268],
            [1, 0.45148398, 0, 0.24854027, 0.10743325, 0.32255035, 0.28851305]]




class Clustering:
    def __init__(self,clustering_train_data):
        self.type = clustering_train_data['type']
        # clustering_train_data['data']是json格式，为了方便实用，在这里直接转换为字典型
        self.data = json.loads(clustering_train_data['data'])


    def deal_with_data(self):
        '''    阶段一：数据准备阶段       '''
        '''          处理数据           '''
        # 将json格式的数据转换为dataframe.再进行归一化处理
        temp_df = pd.DataFrame(self.data)
        # print(temp_df)
        temp_data = np.array(temp_df)
        # 数据归一后聚类效果更准
        final_data = minmax_normaliize(temp_data)
        return final_data

    '''返回node端3 个图'''
    def evalute_cluster_k(self):
        data = self.deal_with_data()
        '''阶段二：聚类算法返回3个图像给用户评估聚类簇数k '''
        def estimate_k(final_data):
            '''评估'''

            # 方法一
            def sse(x):
                '''
                簇内误方差，用来找曲线肘点确定最合适的k
                '''
                # 用来存放设置不同簇数是的sse值
                distortions = []
                for i in range(1, 10):
                    clf = KMeans(n_clusters=i)
                    clf.fit(x)
                    # 获取kmeans sse值
                    distortions.append(clf.inertia_)
                yi_jie_dao = []
                er_jie_dao = []
                for i in range(0, 8):
                    cha = distortions[i + 1] - distortions[i]
                    yi_jie_dao.append(cha)
                # print(yi_jie_dao)
                for j in range(0, 7):
                    cha2 = abs(yi_jie_dao[j + 1] - yi_jie_dao[j])
                    er_jie_dao.append(cha2)
                    # print(er_jie_dao)
                k = er_jie_dao.index(min(er_jie_dao)) + 1

                plt.plot(range(1, 10), distortions, marker='o')
                plt.xlabel("簇数量")
                plt.ylabel("簇内误方差（SSE）")
                plt.show()
                # 保存图像
                # plt.savefig('簇内误方差.png')
                # print(k)
                return distortions

            # 方法二
            def k_silhouette(X, clusters):
                '''
                轮廓系数的折线图确定聚类个数
                '''
                K = range(2, clusters + 1)
                # 构建空列表，用于存储个中簇数下的轮廓系数
                S = []
                for k in K:
                    kmeans = KMeans(n_clusters=k)
                    kmeans.fit(X)
                    labels = kmeans.labels_
                    # 调用字模块metrics中的silhouette_score函数，计算轮廓系数
                    S.append(metrics.silhouette_score(X, labels, metric='euclidean'))

                # 中文和负号的正常显示
                plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
                plt.rcParams['axes.unicode_minus'] = False
                # 设置绘图风格
                plt.style.use('ggplot')
                # 绘制K的个数与轮廓系数的关系
                plt.plot(K, S, 'b*-')
                plt.xlabel('簇的个数')
                plt.ylabel('轮廓系数')
                # 显示图形
                plt.show()
                # plt.savefig('轮廓系数.png')
                return S

            # 方法三
            def gap_statistic(X, B=10, K=range(1, 15), N_init=10):
                '''
                # 计算GAP统计量确定聚类数k
                '''

                # 自定义函数，计算簇内任意两样本之间的欧氏距离
                def short_pair_wise_D(each_cluster):
                    mu = each_cluster.mean(axis=0)
                    Dk = sum(sum((each_cluster - mu) ** 2)) * 2.0 * each_cluster.shape[0]
                    return Dk

                # 计算簇内的Wk值
                def compute_Wk(data, classfication_result):
                    Wk = 0
                    label_set = set(classfication_result)
                    for label in label_set:
                        each_cluster = data[classfication_result == label, :]
                        Wk = Wk + short_pair_wise_D(each_cluster) / (2.0 * each_cluster.shape[0])
                    return Wk

                # 将输入数据集转换为数组
                X = np.array(X)
                # 生成B组参照数据
                shape = X.shape
                tops = X.max(axis=0)
                bots = X.min(axis=0)
                dists = np.matrix(np.diag(tops - bots))
                rands = np.random.random_sample(size=(B, shape[0], shape[1]))
                for i in range(B):
                    rands[i, :, :] = rands[i, :, :] * dists + bots

                # 自定义0元素的数组，用于存储gaps、Wks和Wkbs
                gaps = np.zeros(len(K))
                Wks = np.zeros(len(K))
                Wkbs = np.zeros((len(K), B))
                # 循环不同的k值，
                for idxk, k in enumerate(K):
                    k_means = KMeans(n_clusters=k)
                    k_means.fit(X)
                    classfication_result = k_means.labels_
                    # 将所有簇内的Wk存储起来
                    Wks[idxk] = compute_Wk(X, classfication_result)

                    # 通过循环，计算每一个参照数据集下的各簇Wk值
                    for i in range(B):
                        Xb = rands[i, :, :]
                        k_means.fit(Xb)
                        classfication_result_b = k_means.labels_
                        Wkbs[idxk, i] = compute_Wk(Xb, classfication_result_b)

                # 计算gaps、sd_ks、sk和gapDiff
                gaps = (np.log(Wkbs)).mean(axis=1) - np.log(Wks)
                sd_ks = np.std(np.log(Wkbs), axis=1)
                sk = sd_ks * np.sqrt(1 + 1.0 / B)
                # 用于判别最佳k的标准，当gapDiff首次为正时，对应的k即为目标值
                gapDiff = gaps[:-1] - gaps[1:] + sk[1:]

                # 中文和负号的正常显示
                plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
                plt.rcParams['axes.unicode_minus'] = False
                # 设置绘图风格
                plt.style.use('ggplot')
                # 绘制gapDiff的条形图
                plt.bar(np.arange(len(gapDiff)) + 1, gapDiff, color='steelblue')
                plt.xlabel('簇的个数')
                plt.ylabel('k的选择标准')
                plt.show()

            # 方法四

            def gapStat(data, resf=None, nrefs=10, ks=range(1, 10)):
                '''
                Gap statistics
                '''
                # MC
                start = time.perf_counter()
                shape = data.shape
                if resf == None:
                    x_max = data.max(axis=0)
                    x_min = data.min(axis=0)
                    dists = np.matrix(np.diag(x_max - x_min))
                    rands = np.random.random_sample(size=(shape[0], shape[1], nrefs))
                    for i in range(nrefs):
                        rands[:, :, i] = rands[:, :, i] * dists + x_min
                else:
                    rands = resf
                gaps = np.zeros((len(ks),))
                gapDiff = np.zeros(len(ks) - 1, )
                sdk = np.zeros(len(ks), )
                for (i, k) in enumerate(ks):
                    # 对形成k个群集的一组观察向量执行k均值。
                    # k均值算法将观测值的分类调整为聚类，并更新聚类质心，直到质心的位置在连续迭代中保持稳定为止。
                    # 在算法的这种实现中，质心的稳定性是通过将观测值及其对应质心之间的平均欧几里德距离的变化的绝对值与阈值进行比较来确定的。
                    # 这样便产生了一个将质心映射到代码的codebook，反之亦然。
                    (cluster_mean, cluster_res0) = scipy.cluster.vq.kmeans(data, k)
                    cluster_res = pairwise_distances_argmin(data, cluster_mean)
                    Wk = sum([EuclDist(data[m, :], cluster_mean[cluster_res[m], :]) for m in range(shape[0])])
                    WkRef = np.zeros((rands.shape[2],))
                    for j in range(rands.shape[2]):
                        (kmc, kml) = scipy.cluster.vq.kmeans(rands[:, :, j], k)
                        kml = pairwise_distances_argmin(rands[:, :, j], kmc)
                        WkRef[j] = sum([EuclDist(rands[m, :, j], kmc[kml[m], :]) for m in range(shape[0])])
                    gaps[i] = numpy.log(numpy.mean(WkRef)) - numpy.log(Wk)
                    sdk[i] = np.sqrt((1.0 + nrefs) / nrefs) * np.std(numpy.log(WkRef))
                    if i > 0:
                        gapDiff[i - 1] = gaps[i - 1] - gaps[i] + sdk[i]
                f, (a1, a2) = plt.subplots(2, 1)
                a1.plot(gaps, 'g-o')
                a2.bar(np.arange(len(gapDiff)), gapDiff)
                f.show()
                print(gapDiff)
                return gapDiff

            # 拐点可视化：关注斜率的变化，斜率突然变小且变化缓慢  结果在3，4,5，6
            sse(final_data)
            # 轮廓系数法：最接近1 的k为最合适的聚类k：4,5,8,9,10
            k_silhouette(final_data, 15)
            # 间隔GAP统计量确定聚类数  ：首次出现正值的聚类数（没有＞0出现，但是3，4,6-》往后都很小，6,7最小）
            gap_statistic(final_data)
            # 间隔GAP方法2，使用scipy数学计算模块加速GAP统计量的计算
            gapStat(final_data)
        estimate_k(data)



    '''          训练聚类模型          '''
    '''从node端获取簇数 k 和 模型保存地址'''
    def train_model(self,k,dir_model):
        '''
               k：分类种类数
               x：train_data
               '''
        train_data = self.deal_with_data()
        k = int(k)
        clf = KMeans(n_clusters=k)
        clf.fit(train_data)
        centers = clf.cluster_centers_  # 数据的中心点
        labels = clf.labels_  # 每个数据所述分组
        # print('数据中心点\n', centers)
        # print('数据标签：\n', labels)
        data = []
        for i in range(0, k):
            data.append([])
        for j in range(0, len(labels)):
            k = labels[j]
            data[k].append(train_data[j])
        # print('分类数据：\n', data)
        # 存储模型
        output = open(dir_model,'wb')
        pickle.dump(clf,output)
        output.close()
        return clf
        # # 调出模型代码如下
        # input = open(filename,'rb')
        # clf_2 = pickle.load(input)
        # input.close()


    '''使用模型预测分类结果'''
    def clustering_predict(self,predict_data,dir_model):
        # 调出模型代码如下
        input = open(dir_model,'rb')
        clf = pickle.load(input)
        label = clf.predict(predict_data)
        input.close()
        print(label)
        return label


    '''还额外编写了一个处理标签的函数（当预测数据量很多时方便查看）'''
    def dowith_lables(self):
        '''
        返回按照分类标签分类后的的切片索引情况
        '''
        labels = self.clustering_predict()
        l = len(labels)
        start = []
        end = []
        flag = 1
        start.append(0)
        for i in range(0, l - 1):
            if (labels[i] != labels[i + 1]):
                if (flag % 2 == 0):
                    start.append(i + 1)
                    end.append(i)
                    flag += 1
                else:
                    end.append(i)
                    start.append(i + 1)
                    flag += 1
        end.append(l)
        clf_index = dict(zip(start, end))
        return  clf_index

def main():
    # clustering_train_data = input('请输入json格式数据')
    clustering = Clustering(clustering_train_data)
    clustering.evalute_cluster_k()
    k = input('请输入聚类簇数k：\n')
    dir_model = input('请输入模型保存地址：\n')
    # 训练模型并保存
    clustering.train_model(k,dir_model)
    pre_data = input('请输入待分类数据：\n')
    # 预测数据标签
    clustering.clustering_predict(pre_data,dir_model)



if __name__ == '__main__':
    main()


'''测试聚类模型预测的示例'''
    # a = clf_model.predict([[1, 0.45148398, 0,0.24854027, 0.10743325,0.32255035,0.28851305]])   【3】
    # a1 = clf_model.predict([[1, 0.48699788, 0, 0.7100488, 0.11844756,0.17191174, 0.25296268]]【1】)
    # print(a)
    # print(a1)
