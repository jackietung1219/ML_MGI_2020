from joblib import load
import for_data_single

from PyQt5.QtWidgets import QApplication, QFileDialog, QGridLayout, QLabel, QWidget
from PyQt5 import QtWidgets
import PyQt5.QtCore

import sys
import numpy as np

# version
_version_single = "v_1"

# local model load as variables
model_MLPr_i3_h = load("_train_model/MLPr_i3_h.joblib")
model_MLPr_i3_r = load("_train_model/MLPr_i3_r.joblib")
model_RFr_h = load("_train_model/RFr_h.joblib")
model_RFr_r = load("_train_model/RFr_r.joblib")

# style sheet
widget_style = """
#Form{
    background-color: #FFFFFF;
}

QLabel{
    font: 12pt "Calibri";
}

#groupBox_data_generate, #groupBox_prediction{
    font: 14pt "Calibri";
    background-color: #FFFFFF;
    border: 1px solid black;
    border-radius: 10px;
}

#groupBox_data_generate:title, #groupBox_prediction:title {
    border: 0px solid black;
    border-radius: 10px;
    background-color: #B3D9D9;
}

#pushButton_predict{
    font: 12pt "Calibri";
    background-color: #BEBEBE;
    border: 0px solid black;
    border-radius: 10px;
}

#lineEdit_v, #lineEdit_theta{
    font: 12pt "Calibri";
    border: 1px solid black;
    border-radius: 5px;
}

"""


class TestUI(QWidget):
    """UI class, combined predicting"""

    def __init__(self):
        super().__init__()
        self.ui = for_data_single.Ui_Form()
        self.ui.setupUi(self)

        self.set_stylesheet()
        self._object_init()
        self._button_connect()

    def set_stylesheet(self):
        """Set the style sheet"""
        self.setStyleSheet(widget_style)

    def _object_init(self):
        self.ui.label_version.setText(f"version: {_version_single}")
        self.setWindowTitle("Single Data Prediction")
        self.ui.label_h_predict.setText("")
        self.ui.label_r_predict.setText("")
        self.ui.label_h_real.setText("")
        self.ui.label_r_real.setText("")

    def _button_connect(self):
        """When ever button clicked"""

        def predict():
            """Start predicting"""
            try:
                v, theta = self.ui.lineEdit_v.text(), self.ui.lineEdit_theta.text()
                input_val = np.array((v, theta))
                input_val = input_val.reshape(1, 2)
                print(input_val.shape)
                h_pre = model_RFr_h.predict(input_val)[0]
                r_pre = model_RFr_r.predict(input_val)[0]

                h_real = (float(v) * np.sin(float(theta) * (np.pi / 180))) ** 2 / 19.6
                r_real = (float(v) ** 2) * np.sin(2 * float(theta) * (np.pi / 180)) / 9.8

                self.ui.label_h_predict.setText(f"h = {'%.3f' %h_pre}")
                self.ui.label_r_predict.setText(f"r = {'%.3f' %r_pre}")
                self.ui.label_h_real.setText(f"h = {'%.3f' %h_real}")
                self.ui.label_r_real.setText(f"r = {'%.3f' %r_real}")
            except:
                print("Something wrong in predict()")

        self.ui.pushButton_predict.clicked.connect(predict)


def main():
    # i don't know it either
    app = QApplication(sys.argv)

    # get ui instance
    monitor_w = TestUI()

    # ui show
    monitor_w.show()

    # if gui exit, system exit
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
