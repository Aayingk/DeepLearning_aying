"""
scr曲线你和函数回归
kernel='linear'时，为线性核，C越大分类效果越好，但有可能会过拟合（defaul C=1）
kernel='rbf'时（default），为高斯核，gamma值越小，分类界面越连续；gamma值越大，分类界面越“散”，分类效果越好，但有可能会过拟合。
decision_function_shape='ovr'时，为one v rest，即一个类别与其他类别进行划分，
decision_function_shape='ovo'时，为one v one，即将类别两两之间进行划分，用二分类的方法模拟多分类的结果。
"""

from sklearn import svm
import utils as dd

def svm_train(x_train, y_train):


    # model = svm.SVC(C=0.08, kernel="linear", decision_function_shape="ovr")  # 线性核分类器
    model= svm.SVC(C=0.8, kernel="rbf", gamma=20, decision_function_shape="ovr")  # 高斯核分类器

    model.fit(x_train, y_train.ravel())  # training the svc model,ravel转置

    # print("测试集拟合度:")
    # print(model.score(x_test, y_test))  # 测试集拟合度
    # 验证集预测
    # print("验证集预测值:")
    # print(model.predict(x_test))
    # print("验证集真实值:")
    # print(y_test.ravel())
    return model


def is_exception(model,k,x_test,y_test):
    lable = model.predict(x_test)
    # print("lable",lable)
    l = len(x_test)
    sum = 0
    for i in range(0,l):
        if(lable[i]==y_test[i]):
            sum+=1
    score = sum/l
    # print(score)
    if(score>=k):
        print("当前为PRZ liquid space leak事故工况！")
        return True
    else:
        print("当前为正常工况。")
        return False



if __name__ == "__main__":
    # 读入数据
    concat_data,data = dd.data_read("2_1.csv")
    # 数据集划分
    x_train, x_test, y_train, y_test = dd.data_split(concat_data)

    # svm分类器
    model= svm_train( x_train, y_train)


    # x_test, y_test = dd.return_x_y_test("1_1.csv")
    # x_test, y_test = dd.return_x_y_test("2_1.csv")
    x_test, y_test = dd.return_x_y_test("2_2.csv")
    # # 判断是否为当前事故工况
    is_exception(model,0.9,x_test, y_test)