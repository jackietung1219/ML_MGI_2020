# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'for_testUI_v2.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(511, 302)
        self.groupBox_data_generate = QtWidgets.QGroupBox(Form)
        self.groupBox_data_generate.setGeometry(QtCore.QRect(20, 20, 181, 231))
        self.groupBox_data_generate.setObjectName("groupBox_data_generate")
        self.pushButton_predict = QtWidgets.QPushButton(self.groupBox_data_generate)
        self.pushButton_predict.setGeometry(QtCore.QRect(70, 170, 101, 51))
        self.pushButton_predict.setObjectName("pushButton_predict")
        self.label_v = QtWidgets.QLabel(self.groupBox_data_generate)
        self.label_v.setGeometry(QtCore.QRect(10, 50, 111, 31))
        self.label_v.setObjectName("label_v")
        self.lineEdit_v = QtWidgets.QLineEdit(self.groupBox_data_generate)
        self.lineEdit_v.setGeometry(QtCore.QRect(130, 50, 41, 31))
        self.lineEdit_v.setObjectName("lineEdit_v")
        self.label_theta = QtWidgets.QLabel(self.groupBox_data_generate)
        self.label_theta.setGeometry(QtCore.QRect(10, 109, 111, 31))
        self.label_theta.setObjectName("label_theta")
        self.lineEdit_theta = QtWidgets.QLineEdit(self.groupBox_data_generate)
        self.lineEdit_theta.setGeometry(QtCore.QRect(130, 110, 41, 31))
        self.lineEdit_theta.setObjectName("lineEdit_theta")
        self.groupBox_prediction = QtWidgets.QGroupBox(Form)
        self.groupBox_prediction.setGeometry(QtCore.QRect(230, 20, 271, 271))
        self.groupBox_prediction.setObjectName("groupBox_prediction")
        self.label_predict_value = QtWidgets.QLabel(self.groupBox_prediction)
        self.label_predict_value.setGeometry(QtCore.QRect(10, 150, 141, 31))
        self.label_predict_value.setObjectName("label_predict_value")
        self.label_r_predict = QtWidgets.QLabel(self.groupBox_prediction)
        self.label_r_predict.setGeometry(QtCore.QRect(140, 200, 121, 31))
        self.label_r_predict.setObjectName("label_r_predict")
        self.label_h_real = QtWidgets.QLabel(self.groupBox_prediction)
        self.label_h_real.setGeometry(QtCore.QRect(140, 50, 121, 31))
        self.label_h_real.setObjectName("label_h_real")
        self.label_h_predict = QtWidgets.QLabel(self.groupBox_prediction)
        self.label_h_predict.setGeometry(QtCore.QRect(140, 150, 121, 31))
        self.label_h_predict.setObjectName("label_h_predict")
        self.label_true_value = QtWidgets.QLabel(self.groupBox_prediction)
        self.label_true_value.setGeometry(QtCore.QRect(10, 50, 141, 31))
        self.label_true_value.setObjectName("label_true_value")
        self.label_r_real = QtWidgets.QLabel(self.groupBox_prediction)
        self.label_r_real.setGeometry(QtCore.QRect(140, 96, 121, 31))
        self.label_r_real.setObjectName("label_r_real")
        self.label_version = QtWidgets.QLabel(Form)
        self.label_version.setGeometry(QtCore.QRect(20, 260, 121, 31))
        self.label_version.setObjectName("label_version")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.groupBox_data_generate.setTitle(_translate("Form", "Data Generate"))
        self.pushButton_predict.setText(_translate("Form", "predict"))
        self.label_v.setText(_translate("Form", "v"))
        self.label_theta.setText(_translate("Form", "theta"))
        self.groupBox_prediction.setTitle(_translate("Form", "Prediction"))
        self.label_predict_value.setText(_translate("Form", "Predict Value"))
        self.label_r_predict.setText(_translate("Form", "r"))
        self.label_h_real.setText(_translate("Form", "h"))
        self.label_h_predict.setText(_translate("Form", "h"))
        self.label_true_value.setText(_translate("Form", "True Value"))
        self.label_r_real.setText(_translate("Form", "r"))
        self.label_version.setText(_translate("Form", "version"))
