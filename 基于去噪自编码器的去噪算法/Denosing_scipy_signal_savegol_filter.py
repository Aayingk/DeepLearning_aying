import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal
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


testpoint_arr = ['滑油总管流量','滑油冷却器滑油进口温度','滑油冷却器1号滑油入口压力','滑油冷器冷却水进口流量','滑油冷却器1号冷却水入口温度','滑油冷却器1号冷却水出口温度','滑油总管温度']
short_t_arr = ['F1','TT1','PT1','F2','TT2','TT3','TT4']

df = pd.read_csv('scan_data.csv',encoding='gbk')
print(df)
print(df.columns)
y0 = df[testpoint_arr]
y0.columns = short_t_arr
print(y0)
two_arr = []
for i in range(0,7):
    y1 = y0[short_t_arr[i]]
    y_data = np.array(list(y1))
    l =len(y_data)
    t = np.arange(0,l,1)
    y = scipy.signal.savgol_filter(y_data, 501, 3)
    # 注意这里第二个参数规定是奇数
    two_arr.append(y)
    # 绘图
    plt.plot(t, y_data, "b.-", t, y, "r.-")

    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend(['original data', 'smooth data'])
    plt.grid(True)
    plt.show()
#
# df2 = pd.DataFrame(two_arr,index = short_t_arr).transpose()
#
# df2.to_csv('denoising_501_data.csv')