from sklearn import svm


class SVM_obj:
    def __init__(self,excep_name,C, kernel):
        '''
        :param excep_name: 事故分类：(取值：【"PRZ_lsl","PRZ_vsl","RCS_CL_LOCA_1","RCS_CL_LOCA_2","RCS_HL_LOCA_1","RCS_HL_LOCA_2","SG_2nd_sl","SGTR_60","SGTR_100"】
        :param C: C越大分类效果越好，但有可能会过拟合（defaul C=1）
        :param kernel: 高斯核 {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'}, default='rbf'
        '''
        '''9个事故 的 SVM模型'''
        if (excep_name == "PRZ_lsl"):
            self.PRZ_lsl = SVM()
            self.model = self.PRZ_lsl.build_model(C, kernel)
        if (excep_name == "PRZ_vsl"):
            self.PRZ_vsl = SVM()
            self.model = self.PRZ_vsl.build_model(C, kernel)
        if (excep_name == "RCS_CL_LOCA_1"):
            self.RCS_CL_LOCA_1 = SVM()
            self.model = self.RCS_CL_LOCA_1.build_model(C, kernel)
        if (excep_name == "RCS_CL_LOCA_2"):
            self.RCS_CL_LOCA_2 = SVM()
            self.model = self.RCS_CL_LOCA_2.build_model(C, kernel)
        if (excep_name == "RCS_HL_LOCA_1"):
            self.RCS_HL_LOCA_1 = SVM()
            self.model = self.RCS_HL_LOCA_1.build_model(C, kernel)
        if (excep_name == "RCS_HL_LOCA_2"):
            self.RCS_HL_LOCA_2 = SVM()
            self.model = self.RCS_HL_LOCA_2.build_model(C, kernel)
        if (excep_name == "SG_2nd_sl"):
            self.SG_2nd_sl = SVM()
            self.model = self.SG_2nd_sl.build_model(C, kernel)
        if (excep_name == "SGTR_60"):
            self.SGTR_60 = SVM()
            self.model = self.SGTR_60.build_model(C, kernel)
        if (excep_name == "SGTR_100"):
            self.SGTR_100 = SVM()
            self.model = self.SGTR_100.build_model(C, kernel)

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train.ravel())

    def is_exception(self, k, x_test, y_test):
        lable = self.model.predict(x_test)
        l = len(x_test)
        sum = 0
        for i in range(0, l):
            if (lable[i] == y_test[i]):
                sum += 1
        score = sum / l
        # print(score)
        if (score >= k):
            print("当前为PRZ liquid space leak事故工况！")
            return True
        else:
            print("当前为正常工况。")
            return False

class SVM:
    def build_model(self, C, kernel):
        self.model = svm.SVC(C=C, kernel=kernel, gamma=20, decision_function_shape="ovr")
        return self.model