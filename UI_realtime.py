import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import cv2
from PyQt5 import QtGui
from segment.predict import run
import os
import shutil
import numpy as np

class MainWindow(QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.VBL = QVBoxLayout()

        self.FeedLabel = QLabel()
        self.VBL.addWidget(self.FeedLabel)

        self.CancelBTN = QPushButton("Cancel")
        self.CancelBTN.clicked.connect(self.CancelFeed)
        self.VBL.addWidget(self.CancelBTN)

        self.Worker1 = Worker1()

        self.Worker1.start()
        self.Worker1.ImageUpdate.connect(self.ImageUpdateSlot)
        self.setLayout(self.VBL)


 # Function for updating the image *********************
    def ImageUpdateSlot(self, Image):

        # directory_path = "/Users/suhelkhan/NHRI_woundDetect/PYQT5_Projects/yolov7/seg/results_pred/exp"
        # if os.path.exists(directory_path) and os.path.isdir(directory_path):
        #      try:
        #           shutil.rmtree(directory_path)
        #      except OSError as e:
        #           print(f"Error: {directory_path} : {e.strerror}")
        #      else:
        #           print(f"Directory '{directory_path}' does not exist.")

        self.FeedLabel.setPixmap(QPixmap.fromImage(Image))


    def CancelFeed(self):
        self.Worker1.stop()
        cv2.destroyAllWindows()

#****************** A thread which is responsible for taking each frame and detect the wound
class Worker1(QThread):
    ImageUpdate = pyqtSignal(QImage)
    def run(self):
        self.ThreadActive = True
        Capture = cv2.VideoCapture(0)
        Capture.set(3,800) #set frame width
        Capture.set(4,600) #set frame height
        while self.ThreadActive:
            ret, frame = Capture.read()
            if ret:
                # Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cv2.imwrite("/Users/suhelkhan/NHRI_woundDetect/PYQT5_Projects/yolov7/seg/result.jpg" , frame)
                # img = cv2.imread("/Users/suhelkhan/NHRI_woundDetect/PYQT5_Projects/yolov7/seg/result.jpg")

                directory_path = "/Users/suhelkhan/NHRI_woundDetect/PYQT5_Projects/yolov7/seg/results_pred/exp"
                if os.path.exists(directory_path) and os.path.isdir(directory_path):
                   try:
                     shutil.rmtree(directory_path)
                   except OSError as e:
                     print(f"Error: {directory_path} : {e.strerror}")
                   else:
                     print(f"Directory '{directory_path}' does not exist.")


                run(weights="/Users/suhelkhan/NHRI_woundDetect/PYQT5_Projects/yolov7/seg/yolov7_with_category_30_epochs/weights/best.pt",
                source= "/Users/suhelkhan/NHRI_woundDetect/PYQT5_Projects/yolov7/seg/result.jpg" , project="/Users/suhelkhan/NHRI_woundDetect/PYQT5_Projects/yolov7/seg/results_pred")
                print("Y")
                ConvertToQtFormat = np.array(cv2.imread("/Users/suhelkhan/NHRI_woundDetect/PYQT5_Projects/yolov7/seg/results_pred/exp/result.jpg"))
                FlippedImage = cv2.cvtColor(ConvertToQtFormat, cv2.COLOR_BGR2RGB)
                ConvertToQtFormat = QImage(FlippedImage.data, FlippedImage.shape[1], FlippedImage.shape[0], QImage.Format_RGB888)
                Pic = ConvertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                self.ImageUpdate.emit(Pic)


    def stop(self):
        self.ThreadActive = False
        self.quit()


if __name__ == "__main__":
    App = QApplication(sys.argv)
    Root = MainWindow()
    Root.show()
    sys.exit(App.exec())
