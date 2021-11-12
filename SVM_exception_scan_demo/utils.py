import pandas as pd
import numpy as np
from sklearn import model_selection

normal_state = '1_1.csv'
def data_read(exception_state):
    '''
    :param exception_state: eg:"2_1.csv"
                            PS: 代表事故2的第一个TXT事故数据处理后的csv文件
    :return: 拼接后的包含正常和事故工况的含有标签的dataframe数据
    '''
    normal_df = pd.read_csv(normal_state,encoding="gbk")
    normal_df.insert(normal_df.shape[1],"lable",int(0))
    exception_df = pd.read_csv("data/"+exception_state,encoding="gbk")
    # 为事故数据添加lable标签
    exception_df.insert(exception_df.shape[1],"lable",int(1))
    concat_data = pd.concat([normal_df,exception_df],axis=0)
    # print("normal_df",normal_df)
    # print("exception_state",exception_df)
    # print("concat_data",concat_data)
    data = concat_data.values
    # print("data",data)
    return concat_data,data

def data_split(data):
    """
    :param data:读入列表
    :return:数据集划分为训练集和验证集,x是特征y是类别标志
    """
    x, y = np.split(data, (24,), axis=1)
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        x, y, random_state=1, train_size=0.7)
    x_train = x_train.values
    # print("x_train",x_train)
    x_test = x_test.values
    # print("x_test",x_test)
    y_train = y_train.values
    # print("y_train",y_train)
    y_test = y_test.values
    # print("y_test",y_test)
    return x_train, x_test, y_train, y_test

def return_x_y_test(exception_state):
    exception_df = pd.read_csv("data/" + exception_state, encoding="gbk")
    # 为事故数据添加lable标签
    exception_df.insert(exception_df.shape[1], "lable", int(1))
    x_test, y_test = np.split(exception_df.values, (24,), axis=1)
    return x_test,y_test