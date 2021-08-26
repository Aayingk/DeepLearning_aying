from __future__ import print_function
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller as ADF
from statsmodels.stats.diagnostic import acorr_ljungbox
import matplotlib.pyplot as plot

'''
时序数据的重要模型——AR  +  I  +  MA 模型知识点
1，要求平稳性：要求在未来一段时间仍能顺着现有的形态‘惯性’的延续下去。要求序列的方差和均值不发生明显变化,。
2，一阶差分法：在原始时间序列上计在t和t-1时刻的差
3，二阶差分法：在一阶差分的基础上做差分    ps：做差分越做数据越平稳，在做ARIMA 前进行数据差分处理可以保证数据的平稳性
4，自回归模型：是指我只有我自身的数据，不了解当前变量和其他变量的关系，只能使用自身的历史数据对自身进行预测。
5，p阶自回归是今天和前p天都有关，
6，注意 p的取值是有专业的评估标准的，根据这个标准确定p是最科学的
7.arima模型要求平稳性，只有平稳数据才能通过自身的李历史数据对数据惊进行预测，需要计算一下自相关系数，如果小于0.5，不宜采用arima

'''
'''
    ARIMA模型预测首先要确定参数p， q，d
    p：自回归模型的阶数
    q：移动平均的阶数
    d：一般取 1,2,3 确定是几阶差分
'''


'''  在正式交付版本中没有这部分，这部分的作用在于给出用户给出的数据参考格式  '''
df = pd.read_csv(open('denoising_501_data.csv')).iloc[0:3001]
data = df[['TT4']]
plt.scatter(np.arange(0, len(data), 1),data)
plt.show()
df2 = pd.read_csv(open('denoising_501_data.csv')).iloc[3000:4001]
data2 = df2[['TT4']]
plt.scatter(np.arange(0, len(data2), 1),data2)
plt.show()

def arima(data,n):
    '''        阶段一：平稳性检验        '''
    # 检验平稳性：ADF(data)输出的第二个值是p值，当p>0.05 时，接受原假设，存在单位根则无需差分（实际就是保证数据不要是单调的）
    # 否则（p<0.05 则进行差分处理）
    # print(ADF(data))
    # 输出
    # (-4.88153569914598, 3.787829044144883e-05, 3, 2997, {'1%': -3.4325338204130245, '5%': -2.862504870605232, '10%': -2.5672836260495844}, -24859.403042659203)
    '''           差分处理            '''
    # '''         该样本数据无需差分         '''
    # d_data = data.diff().dropna()
    # d_data.columns = ['一阶差分']
    # plt.scatter(np.arange(0, len(d_data), 1),d_data)
    # plt.show()
    '''       阶段二 ：随机性检验       '''
    # print(acorr_ljungbox(data,lags=1))
    # (array([2997.85077116]), array([0.]))
    # 0.<0.05 拒绝原假设，所以一阶差分后的序列不是随机的


    '''       阶段三 ：确定 p 和 q 的取值         '''
    from statsmodels.tsa.arima_model import ARIMA
    tmp = []
    for p in range(4):
        for q in range(4):
            try:
                tmp.append([ARIMA(data, (p, 1, q)).fit().bic, p, q])
            except:
                tmp.append([None, p, q])
    tmp = pd.DataFrame(tmp, columns=['bic', 'p', 'q'])
    print(tmp[tmp['bic'] == tmp['bic'].min()])
    print(tmp['p'])

    '''          建模预测            '''
    # p d q
    model = ARIMA(data, (2, 0, 1)).fit()
    model.summary()

    yp = model.forecast(n)  # 预测未来几个单位的数据
    print(yp[0])
    plt.scatter(np.arange(0, len(yp[0]), 1),yp[0])
    plt.show()
    return yp[0]

if __name__ == '__main__':
    arima(data,1000)