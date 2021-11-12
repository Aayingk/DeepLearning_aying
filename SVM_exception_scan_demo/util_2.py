import pandas as pd
import numpy as np

columns = ["热功率","电功率","主回路1冷却剂流量","主回路2冷却剂流量","主回路1热管段冷却剂温度","主回路2热管段冷却剂温度","主回路1冷管段冷却剂温度",
    "主回路2冷管段冷却剂温度","主回路1蒸汽发生器二次侧压力","主回路2蒸汽发生器二次侧压力","主回路1蒸汽发生器水位","主回路2蒸汽发生器水位",
    "主回路1蒸汽发生器给水流量","主回路2蒸汽发生器给水流量","未使用1","未使用2","主回路1蒸汽发生器给水温度","主回路2蒸汽发生器给水温度",
    "未使用3","未使用4","主回路1蒸汽发生器蒸汽出口流量","主回路2蒸汽发生器蒸汽出口流量","主回路1蒸汽发生器蒸汽出口温度","主回路2蒸汽发生器蒸汽出口温度",
    "稳压器压力","稳压器水位","未使用5","稳压器蒸汽空间温度","稳压器水空间温度"]

final_columns = ["热功率","电功率","主回路1冷却剂流量","主回路2冷却剂流量","主回路1热管段冷却剂温度","主回路2热管段冷却剂温度","主回路1冷管段冷却剂温度",
    "主回路2冷管段冷却剂温度","主回路1蒸汽发生器二次侧压力","主回路2蒸汽发生器二次侧压力","主回路1蒸汽发生器水位","主回路2蒸汽发生器水位",
    "主回路1蒸汽发生器给水流量","主回路2蒸汽发生器给水流量","主回路1蒸汽发生器给水温度","主回路2蒸汽发生器给水温度",
    "主回路1蒸汽发生器蒸汽出口流量","主回路2蒸汽发生器蒸汽出口流量","主回路1蒸汽发生器蒸汽出口温度","主回路2蒸汽发生器蒸汽出口温度",
    "稳压器压力","稳压器水位","稳压器蒸汽空间温度","稳压器水空间温度"]


'''处理单独的TXT文件返回 dataframe类型的数据'''
def txt_to_dataframe(file_path):
    '''
    :param file_path: txt文件的最终路径 PS：注意尽量写  / 代替 \
    :return: dataframe 类型的 2-dim
    '''
    file_path = file_path.replace("\\", "/")
    f = open(file_path,"r")
    data= f.readlines()
    l = []
    a = len(data)
    for i in range(a):
        temp_l = []
        b = len(data[0].split(","))
        for j in range(b):
            temp_l.append(float(data[i].split(",")[j]))
        l.append(temp_l)
    df = pd.DataFrame(l,columns=columns)
    df1 = df[final_columns]
    df2 = minmax_normaliize(df1)
    df2.to_csv("2_2.csv")
    return df2




def minmax_normaliize(data0):
    columns = data0.columns
    data = np.array(data0)
    l = data0.shape[0]
    print(l)
    for  i in range(0,l):
        # print('data[i]',data[i])
        min  = np.amin(data[i])
        # print('mu',mu)
        max = np.amax(data[i])
        # print('std',std)
        data[i] = (data[i] - min)/(max-min)
    data = pd.DataFrame(data,columns=columns)
    return data


if __name__ == '__main__':
    # eg:   "C:/Users/ASUS/Desktop/Original Data/power decreasing/target200 rate30.txt"
    txt_to_dataframe("C:/Users/ASUS/Desktop/Original Data/PRZ liquid space leak/PRZ liquid space leak 0.3.txt")

























