import sys
import cv2

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import yoloCustomObjectDetection as yolo


class Ui_MainWindow(object):
    #Set the construction of ui and register functions for components
    def setupUi(self, MainWindow):
        #logical parameter definition
        self.cap = cv2.VideoCapture(0)

        #checkMaskTimer 
        self.checkMaskTimer = QTimer()
        self.checkMaskTimer.timeout.connect(self.checkMask)
        self.playerTimer = QTimer()
        self.playerTimer.timeout.connect(self.getImage)
        self.detector = yolo.Detection(capture_index=0, model_name='best.pt')

        self.plotFPS = False
        self.plotBox = False
        self.currentLabel = ""


        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(648, 600)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        
        MainWindow.setFont(font)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.detect_result_label = QtWidgets.QLabel(self.centralwidget)
        self.detect_result_label.setGeometry(QtCore.QRect(220,320,191,50))
        self.detect_result_label.setFont(font)
        self.detect_result_label.setAlignment(Qt.AlignCenter)
        self.detect_result_label.setText("")
        self.detect_result_label.setObjectName("detect_result_label")

        self.video_output = QtWidgets.QLabel(self.centralwidget)
        self.video_output.setGeometry(QtCore.QRect(120, 0, 400, 300))
        self.video_output.setStyleSheet("")
        self.video_output.setText("")
        self.video_output.setObjectName("video_output")
        self.checkBox = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox.setGeometry(QtCore.QRect(220, 460, 131, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.checkBox.setFont(font)
        self.checkBox.setObjectName("checkBox")
        self.start_detect = QtWidgets.QPushButton(self.centralwidget)
        self.start_detect.setGeometry(QtCore.QRect(220, 390, 191, 71))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(22)
        self.start_detect.setFont(font)
        self.start_detect.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.DefaultContextMenu)
        self.start_detect.setStyleSheet("")
        self.start_detect.setObjectName("start_detect")
        font.setFamily("Arial")
        font.setPointSize(16)
        self.checkBox_2 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_2.setFont(font)
        self.checkBox_2.setGeometry(QtCore.QRect(220, 500, 111, 31))
        self.checkBox_2.setObjectName("checkBox_2")
        self.stop_detect_button = QtWidgets.QPushButton(self.centralwidget)
        self.stop_detect_button.setGeometry(QtCore.QRect(250,540,120,30))
        self.stop_detect_button.setObjectName("stop_detect_button")
        self.stop_detect_button.setFont(font)
        

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 648, 30))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        #binding buttons and checkboxes
        self.start_detect.clicked.connect(self.startDetect)
        self.stop_detect_button.clicked.connect(self.stopDetect)
        self.checkBox.stateChanged.connect(self.ifPlotBox)
        self.checkBox_2.stateChanged.connect(self.ifPlotFPS)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.checkBox.setText(_translate("MainWindow", "Plot Box"))
        self.start_detect.setText(_translate("MainWindow", "Start Detect"))
        self.checkBox_2.setText(_translate("MainWindow", "Plot FPS"))
        self.stop_detect_button.setText(_translate("MainWindow","Stop Detect"))

    def startDetect(self):
        self.playerTimer.start(0)
        self.checkMaskTimer.start(500)

    def stopDetect(self):
        self.playerTimer.stop()
        self.checkMaskTimer.stop()

    def ifPlotBox(self):
        if self.checkBox.checkState()==2:
            self.plotBox=True
        else:
            self.plotBox=False
    
    def ifPlotFPS(self):
        if self.checkBox_2.checkState()==2:
            self.plotFPS=True
        else:
            self.plotFPS=False

    def checkMask(self):
        self.detect_result_label.setText(self.currentLabel)

    #Get result frame from the class yoloCustomObjectDetection
    def getImage(self):
        assert self.cap.isOpened()
        frame,self.currentLabel = self.detector(self.cap,self.plotBox,self.plotFPS)
        
        show = cv2.resize(frame,(400,300))
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
        self.video_output.setPixmap(QtGui.QPixmap.fromImage(showImage))
    


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    styleSheet = """
        QWidget{
            background:#262d37;
            color:white;
        }
        #detect_result_label{
            text-align:center;
            font-size:20px;
            border:3px solid white;
            border-radius:5px;
        }
        #start_detect{
            background:#0577a8;
            border:3px solid #dadada ;
            border-radius:3px;
        }
        #start_detect:hover{
            background:#05aab0;
        }
        #video_output{
            border:6px solid #fff ;
            border-radius:8px;
        }
        QCheckBox:checked{
            color:#ff4c4c;
        }
        #stop_detect_button{
            background:#ff1e1e;
            border:3px solid #aa1212;
            border-radius:3px;
        }
        #stop_detect_button:hover{
            background:#fc6565;
        }
    """

    app.setStyleSheet(styleSheet)
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec())
