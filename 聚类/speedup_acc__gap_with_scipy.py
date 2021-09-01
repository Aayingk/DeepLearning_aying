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
# print(clustering_train_data)

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

    def evalute_cluster_k(self):
        data = self.deal_with_data()
        '''    阶段二：聚类算法返回3个图像给用户评估聚类簇数k    '''
        def estimate_k(final_data):
            '''评估'''
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
                        (kmc,kml) = scipy.cluster.vq.kmeans(rands[:,:,j], k)
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
                end  = time.perf_counter()
                print("running time:%s Seconds" % (end - start))
                return gaps, gapDiff

            gapStat(final_data)
        estimate_k(data)





def main():
    clustering = Clustering(clustering_train_data)
    clustering.evalute_cluster_k()
    k = input('请输入聚类簇数k：\n')




if __name__ == '__main__':
    main()

