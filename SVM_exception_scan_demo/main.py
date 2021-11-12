import utils as dd
import SVM_object as s


if __name__ == '__main__':
    # 拼接2_1的事故数据和1_1的正常数据 得到未划分的训练数据concat_data
    concat_data, data = dd.data_read("2_1.csv")

    # 数据集划分
    x_train, x_test, y_train, y_test = dd.data_split(concat_data)

    # 测试数据（单纯的正常数据或事故数据）
    x_test2, y_test2 = dd.return_x_y_test("1_1.csv")       # 正常数据
    # x_test2, y_test2 = dd.return_x_y_test("2_1.csv")     # 事故数据
    # x_test2, y_test2 = dd.return_x_y_test("2_2.csv")     # 事故数据

    # 实例化事故类型为"PRZ_lsl"的SVM分类模型
    # 参数1的取值，代表那种事故的svm模型 【"PRZ_lsl","PRZ_vsl","RCS_CL_LOCA_1","RCS_CL_LOCA_2","RCS_HL_LOCA_1","RCS_HL_LOCA_2","SG_2nd_sl","SGTR_60","SGTR_100"】
    svm =s.SVM_obj("PRZ_lsl",0.8,"rbf")
    #训练模型
    svm.train(x_train,y_train)

    # # 判断是否为当前事故工况
    svm.is_exception(0.9, x_test2, y_test2)