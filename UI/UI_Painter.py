# -*- coding: utf-8 -*-
# @Time    : 2019/9/28 15:32
# @Author  : zwenc
# @File    : UI_Painter.py

import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QMessageBox, QMainWindow, QGridLayout)
from PyQt5.QtGui import (QPainter, QPen, QImage)
from PyQt5.QtCore import Qt
import cv2
import matplotlib.pyplot as plt
from log.log_output import log
import numpy as np

class Painter(QWidget):
    def __init__(self,width,heigh):
        super(Painter, self).__init__()
        self.image = QImage(width, heigh, QImage.Format_RGB32)  # 图层0  只作图，是矢量信息   可以返回
        self.image.fill(0xffffff)
        self.image1 = self.image.copy()                     # 图层1  保存处理的特效，特效放在这一层，可以返回
        self.image2 = self.image.copy()                     # 图层2  识别框图，识别框图放在这一层

        self.disImageIndex = 0        # 可以是0,1,2
        self.returnImageIndex = 0     # 一般只return 0 和 1

        self.paintSize = 10

        # setMouseTracking设置为False，否则不按下鼠标时也会跟踪鼠标事件
        self.setMouseTracking(False)

        self.pos_xy = []

    def paintEvent(self, event):
        if self.disImageIndex == 0:
            self.image.fill(0xffffff)
            painter = QPainter(self.image)
            pen = QPen(Qt.black, self.paintSize, Qt.SolidLine)
            painter.setPen(pen)

            if len(self.pos_xy) > 1:
                point_start = self.pos_xy[0]
                for pos_tmp in self.pos_xy:
                    point_end = pos_tmp

                    if point_end == (-1, -1):
                        point_start = (-1, -1)
                        continue
                    if point_start == (-1, -1):
                        point_start = point_end
                        continue

                    painter.drawLine(point_start[0], point_start[1], point_end[0], point_end[1])
                    point_start = point_end

        painter = QPainter()
        painter.begin(self)
        if self.disImageIndex == 0:
            painter.drawImage(0, 0, self.image)
        elif self.disImageIndex == 1:
            painter.drawImage(0, 0, self.image1)
        else:
            painter.drawImage(0, 0, self.image2)

        painter.end()

    def mouseMoveEvent(self, event):
        self.disImageIndex = 0
        self.returnImageIndex = 0
        # 中间变量pos_tmp提取当前点
        pos_tmp = (event.pos().x(), event.pos().y())
        # pos_tmp添加到self.pos_xy中
        self.pos_xy.append(pos_tmp)

        self.update()

    def mouseReleaseEvent(self, event):
        pos_test = (-1, -1)
        self.pos_xy.append(pos_test)

        self.update()

    def clearWidget(self):
        self.disImageIndex = 0
        self.returnImageIndex = 0

        self.image = QImage(810, 560, QImage.Format_RGB32)
        self.image.fill(0xffffff)

        self.image1 = self.image.copy()
        self.image2 = self.image.copy()
        self.pos_xy = []

        self.update()

    def getPixMap(self):
        if self.returnImageIndex == 0:
            self.image.save(".image.jpg", "JPG")
            log.info_out("生成 .image.jpg 临时文件供上层调用")
            return ".image.jpg"
        else:
            self.image1.save(".image1.jpg", "JPG")
            log.info_out("生成 .image1.jpg 临时文件供上层调用")
            return ".image1.jpg"

    def paintLine(self, x, y, x1, y1):
        if self.disImageIndex != 2:
            if self.returnImageIndex == 0:
                self.image2 = self.image.copy()
            else:
                self.image2 = self.image1.copy()

        self.disImageIndex = 2

        painter = QPainter(self.image2)
        pen = QPen(Qt.red, 1, Qt.SolidLine)
        painter.setPen(pen)
        painter.drawRect(x, y, x1, y1)

        self.update()

    def ProcessPic(self, mode):
        if self.returnImageIndex == 0:
            fileName= ".image.jpg"
            self.image.save(fileName,"JPG")
        else:
            fileName= ".image1.jpg"
            self.image1.save(fileName,"JPG")

        temp = plt.imread(fileName)
        kernel = np.ones((3, 3), np.uint8)

        if mode == 0:
            temp = cv2.dilate(temp,kernel)
            temp = cv2.erode(temp,kernel)

        elif mode == 1:
            temp = cv2.dilate(temp,kernel)
        else:
            temp = cv2.erode(temp,kernel)

        plt.imsave(fileName,temp)

        self.image1 = QImage(fileName)
        self.disImageIndex = 1
        self.returnImageIndex = 1

        self.update()

    def mergeTemp2Image(self):
        log.info_out("功能闲置 不写")
        self.update()

    def withDraw(self):
        if len(self.pos_xy) == 1:
            return

        self.pos_xy.pop()
        while self.pos_xy.pop() != (-1,-1):
            if len(self.pos_xy) == 0:
                break

        self.pos_xy.append((-1,-1))

        self.disImageIndex = 0
        self.returnImageIndex = 0

        self.update()

    def disImage(self):
        self.disImageIndex = 0
        self.returnImageIndex = 0

        self.update()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Painter()
    ex.show()
    sys.exit(app.exec_())