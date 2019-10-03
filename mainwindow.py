# -*- coding: utf-8 -*-
# @Time    : 2019/9/28 15:37
# @Author  : zwenc
# @File    : mainwindow.py
import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QMessageBox, QMainWindow, QGridLayout)
from PyQt5 import QtGui
import cv2
import matplotlib.pyplot as plt
from UI.ui_mainWindow import Ui_MainWindow
from UI.UI_Painter import Painter
from imageProcess import ImageProcess
from net.Net import Net
import torch
import numpy as np
from log.log_output import log

class mainWindow(QMainWindow,Ui_MainWindow):
    def __init__(self):
        super(mainWindow,self).__init__()
        self.setupUi(self)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("UI/ico/windows.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.setWindowIcon(icon)
        self.setWindowTitle("手写数字识别")
        self.paint = Painter(700,500)
        self.gridLayout.addWidget(self.paint)
        self.net = Net()
        self.net.load_state_dict(torch.load("net/parameters.pt"))

        self.image_spit = None
        self.spinBox_px.setValue(10)

        with open("UI/css/mainwindow.css","r") as file_css:
            self.setStyleSheet(file_css.read())

    def button_spit_num(self,bool):
        image = self.paint.getPixMap()
        log.info_out(image)

        self.image_spit = ImageProcess(image,self.spit_Box.currentIndex() + 1)

        if len(self.image_spit) == 0:
            QMessageBox.warning(self, "waring", "无法正确分割数字，或者无数字")
            return

        point = self.image_spit.get_rect_point()

        for x, y, x1, y1 in point:
            self.paint.paintLine(x, y, x1 - x, y1 - y)

    def button_clear(self,bool):
        self.paint.clearWidget()
        self.lineEdit.clear()

    def button_recognize(self,bool):
        image = self.paint.getPixMap()
        self.image_spit = ImageProcess(image,self.spit_Box.currentIndex() + 1)
        if len(self.image_spit) == 0:
            QMessageBox.warning(self, "waring", "无法正确分割数字，或者无数字")
            return

        num = []
        for index ,(image) in enumerate(self.image_spit):
            # 自动膨胀和腐蚀
            if self.AutoED.isChecked():
                kernel = np.ones((2, 2), np.uint8)
                image = cv2.dilate(image,kernel)
                kernel = np.ones((1, 1), np.uint8)
                image = cv2.erode(image,kernel)

            if self.BinProcess.isChecked():
                image[image[:,:] < 100] = 0
                image[image[:,:] > 100] = 255

            # 显示测试图片预览
            if self.StudyPreView.isChecked():
                plt.figure(index)
                plt.imshow(image,cmap="gray")
                plt.show()

            # 数据转化为Torch格式
            data = torch.tensor([[image]]).float()

            # 开始识别
            out = self.net(data)

            # 找到最大值
            num.append(torch.max(out.data, 1)[1][0].item())

        # text控件显示结果
        s = ""
        for i in num:
            s = s + str(i)

        self.lineEdit.setText(s)

    def button_withdraw(self,bool):
        self.paint.withDraw()

    def button_clear_alltemp(self,bool):
        self.paint.disImage()

    def button_writeTemp(self,bool):
        self.paint.mergeTemp2Image()

    def button_preView(self,bool):
        self.paint.ProcessPic(self.YH_Box.currentIndex())

    def spin_num_change(self, num):
        self.paint.paintSize = num
        self.paint.update()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = mainWindow()
    ex.show()
    sys.exit(app.exec_())