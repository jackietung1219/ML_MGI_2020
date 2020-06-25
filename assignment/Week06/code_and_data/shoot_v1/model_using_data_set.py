import sys
import os

# for packing
if hasattr(sys, 'frozen'):
    os.environ['PATH'] = sys._MEIPASS + ";" + os.environ['PATH']

from joblib import load

from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog

from plotting import data_visualization
import for_data_set

# version
_version_set = "v_1"

# style sheet
widget_style = """
#Form{
    background-color: #FFFFFF;
}

QLabel{
    font: 12pt "Calibri";
}

#groupBox_model_choose, #groupBox_data_generate{
    font: 16pt "Calibri";
    background-color: #FFFFFF;
    border: 1px solid black;
    border-radius: 10px;
}

#groupBox_model_choose:title, #groupBox_data_generate:title {
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

#toolButton_filedialog_h, #toolButton_filedialog_r{
    font: 12pt "Calibri";
    border: 0px solid black;
    border-radius: 5px;
    background-color: #BEBEBE;
}

#lineEdit_h_model, #lineEdit_r_model{
    font: 12pt "Calibri";
    border: 1px solid black;
    border-radius: 5px;
}

#lineEdit_v_s, #lineEdit_v_e, #lineEdit_t_s, #lineEdit_t_e, #lineEdit_sample_n{
    font: 12pt "Calibri";
    border: 1px solid black;
    border-radius: 5px;
}
"""


class TestUI(QWidget):
    """UI class"""

    def __init__(self):
        super().__init__()
        self.ui = for_data_set.Ui_Form()
        self.ui.setupUi(self)

        self.plotting = Plotting()
        self.set_stylesheet()
        self._object_init()
        self._button_connect()

    def set_stylesheet(self):
        """Set the style sheet"""
        self.setStyleSheet(widget_style)

    def _object_init(self):
        self.ui.label_version.setText(f"version: {_version_set}")
        self.setWindowTitle("Data Set Prediction")

    def _button_connect(self):
        """When ever button clicked"""

        def button_browse_h_model():
            v1, v2 = QFileDialog.getOpenFileName(self,
                                                 "Select one file to open",
                                                 "./",
                                                 "All Files (*);;")
            self.ui.lineEdit_h_model.setText(v1)

        def button_browse_r_model():
            v1, v2 = QFileDialog.getOpenFileName(self,
                                                 "Select one file to open",
                                                 "./",
                                                 "All Files (*);;")
            self.ui.lineEdit_r_model.setText(v1)

        self.ui.pushButton_predict.clicked.connect(self._button_predict)
        self.ui.toolButton_filedialog_h.clicked.connect(button_browse_h_model)
        self.ui.toolButton_filedialog_r.clicked.connect(button_browse_r_model)

    def _button_predict(self):
        """Start predicting"""
        model_h = load(self.ui.lineEdit_h_model.text())
        model_r = load(self.ui.lineEdit_r_model.text())

        velocity_init = float(self.ui.lineEdit_v_s.text())
        velocity_end = float(self.ui.lineEdit_v_e.text())
        theta_init = float(self.ui.lineEdit_t_s.text())
        theta_end = float(self.ui.lineEdit_t_e.text())
        sample_amount = int(self.ui.lineEdit_sample_n.text())

        self.plotting.boundary_param_predict = (velocity_init, velocity_end, sample_amount), (
            theta_init, theta_end, sample_amount)
        self.plotting.models = (model_h, model_r)
        self.plotting.plot3D()


class Plotting():
    def __init__(self):
        self.boundary_param_predict = (None, None)
        self.models = (None, None)
        self.data_fitting = None

    def plot3D(self):
        """Start plotting"""
        data_visualization(self.boundary_param_predict, self.models, self.data_fitting)


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
