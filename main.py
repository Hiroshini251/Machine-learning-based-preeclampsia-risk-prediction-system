from PyQt5 import QtCore, QtGui, QtWidgets
import pandas as pd #for reading dataset
import numpy as np # array handling functions

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(961, 616)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(230, 70, 451, 51))
        self.label.setObjectName("label")
        self.AGE = QtWidgets.QLabel(self.centralwidget)
        self.AGE.setGeometry(QtCore.QRect(290, 160, 61, 21))
        self.AGE.setObjectName("AGE")
        self.SystolicBP = QtWidgets.QLabel(self.centralwidget)
        self.SystolicBP.setGeometry(QtCore.QRect(280, 210, 161, 21))
        self.SystolicBP.setObjectName("SystolicBP")
        self.DiastolicBP = QtWidgets.QLabel(self.centralwidget)
        self.DiastolicBP.setGeometry(QtCore.QRect(280, 270, 161, 21))
        self.DiastolicBP.setObjectName("DiastolicBP")
        self.BS = QtWidgets.QLabel(self.centralwidget)
        self.BS.setGeometry(QtCore.QRect(280, 330, 161, 21))
        self.BS.setObjectName("BS")
        self.Bodytemperature = QtWidgets.QLabel(self.centralwidget)
        self.Bodytemperature.setGeometry(QtCore.QRect(280, 380, 161, 21))
        self.Bodytemperature.setObjectName("Bodytemperature")
        self.HeartRate = QtWidgets.QLabel(self.centralwidget)
        self.HeartRate.setGeometry(QtCore.QRect(280, 430, 161, 21))
        self.HeartRate.setObjectName("HeartRate")
        self.Predict = QtWidgets.QPushButton(self.centralwidget)
        self.Predict.setGeometry(QtCore.QRect(280, 490, 111, 23))
        self.Predict.setObjectName("Predict")
        self.PREDICTOUTPUT = QtWidgets.QLabel(self.centralwidget)
        self.PREDICTOUTPUT.setGeometry(QtCore.QRect(510, 490, 151, 20))
        self.PREDICTOUTPUT.setFrameShape(QtWidgets.QFrame.Box)
        self.PREDICTOUTPUT.setText("")
        self.PREDICTOUTPUT.setObjectName("PREDICTOUTPUT")
        self.HeartRateinput = QtWidgets.QLineEdit(self.centralwidget)
        self.HeartRateinput.setGeometry(QtCore.QRect(510, 430, 151, 20))
        self.HeartRateinput.setObjectName("HeartRateinput")
        self.Bodytemperatureinput = QtWidgets.QLineEdit(self.centralwidget)
        self.Bodytemperatureinput.setGeometry(QtCore.QRect(510, 380, 151, 20))
        self.Bodytemperatureinput.setObjectName("Bodytemperatureinput")
        self.BSinput = QtWidgets.QLineEdit(self.centralwidget)
        self.BSinput.setGeometry(QtCore.QRect(510, 320, 151, 20))
        self.BSinput.setObjectName("BSinput")
        self.DiastolicBPinput = QtWidgets.QLineEdit(self.centralwidget)
        self.DiastolicBPinput.setGeometry(QtCore.QRect(510, 260, 151, 20))
        self.DiastolicBPinput.setObjectName("DiastolicBPinput")
        self.SystolicBPinput = QtWidgets.QLineEdit(self.centralwidget)
        self.SystolicBPinput.setGeometry(QtCore.QRect(510, 210, 151, 20))
        self.SystolicBPinput.setObjectName("SystolicBPinput")
        self.ageinput = QtWidgets.QLineEdit(self.centralwidget)
        self.ageinput.setGeometry(QtCore.QRect(510, 160, 151, 20))
        self.ageinput.setObjectName("ageinput")
        self.TABLETOUTPUT = QtWidgets.QLabel(self.centralwidget)
        self.TABLETOUTPUT.setGeometry(QtCore.QRect(510, 540, 151, 20))
        self.TABLETOUTPUT.setFrameShape(QtWidgets.QFrame.Box)
        self.TABLETOUTPUT.setText("")
        self.TABLETOUTPUT.setObjectName("TABLETOUTPUT")
        self.TABLET = QtWidgets.QLabel(self.centralwidget)
        self.TABLET.setGeometry(QtCore.QRect(280, 540, 161, 21))
        self.TABLET.setObjectName("TABLET")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 961, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.Predict.clicked.connect(self.machinelearning)

    def machinelearning(self):
        try:
            dataset1 = pd.read_csv("Maternal Health Risk Data Set.csv")#reading dataset
            #print(dataset) # printing dataset

            x = dataset1.iloc[:,:-1].values #locating inputs
            y = dataset1.iloc[:,-1].values #locating outputs

            #printing X and Y
            print("x=",x)
            print("y=",y)
        except Exception as t:
            print(t)

        from sklearn.model_selection import train_test_split # for splitting dataset
        x_train,x_test,y_train,y_test = train_test_split(x ,y, test_size = 0.25 ,random_state = 0)
        #printing the spliited dataset
        print("x_train=",x_train)
        print("x_test=",x_test)
        print("y_train=",y_train)
        print("y_test=",y_test)
        #importing algorithm
        from sklearn.ensemble import RandomForestClassifier  
        classifier= RandomForestClassifier(n_estimators= 10, criterion="entropy")  
        print(classifier.fit(x_train, y_train))#trainig Algorithm #Y=B1X1+B2X2+B3X3....BNXN
        y_pred=classifier.predict(x_test) #testing model
        print("y_pred",y_pred) # predicted output
        print("Testing Accuracy")
        from sklearn import metrics
        #Model Accuracy, how often is the classifier correct?
        print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

        Age = self.ageinput.text()
        SystolicBP = self.SystolicBPinput.text()
        DiastolicBP = self.DiastolicBPinput.text()
        BS = self.BSinput.text()
        Bodytemperature = self.Bodytemperatureinput.text()
        HeartRate= self.HeartRateinput.text()

        pre = classifier.predict([[Age,SystolicBP,DiastolicBP,BS,Bodytemperature,HeartRate]])
        print(pre)

        output = pre
        try:
            self.PREDICTOUTPUT.setText(str(output))
            if output== "high risk":
               a1 = "81 mg/d asprin tablet 30 days,"
               print(a1)
               self.TABLETOUTPUT.setText(str(a1))

            elif output== "mid risk":
               a2 = "51 mg/d asprin tablet "
               print(a2)
               self.TABLETOUTPUT.setText(str(a2))

            elif output== "low risk":
               a3 = "32 mg/d asprin tablet"
               print(a3)
               self.TABLETOUTPUT.setText(str(a3))

        except Exception as e:
            print(e)


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:17pt; font-weight:600; font-style:arial black;\">PREECLAMPSIA PREDICTION SYSTEM</span></p></body></html>"))
        self.AGE.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:14pt; font-style:times new roman;\">AGE</span></p></body></html>"))
        self.SystolicBP.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:14pt; font-style:times new roman;\">SystolicBP LEVEL</span></p></body></html>"))
        self.DiastolicBP.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:14pt; font-style:times new roman;\">DiastolicBP</span></p></body></html>"))
        self.BS.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:14pt; font-style:times new roman;\">BS</span></p></body></html>"))
        self.Bodytemperature.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:14pt; font-style:times new roman;\">Body Temperature</span></p></body></html>"))
        self.HeartRate.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:14pt; font-style:times new roman;\">Heart Rate</span></p></body></html>"))
        self.Predict.setText(_translate("MainWindow", "PREDICT"))
        self.TABLET.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:14pt; font-style:times new roman;\">TABLET</span></p></body></html>"))


if _name_ == "_main_":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
