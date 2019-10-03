# -*- coding: utf-8 -*-
# @Time    : 2019/9/28 15:54
# @Author  : zwenc
# @File    : imageProcess.py

import cv2
from log.log_output import log
import numpy as np
import matplotlib.pyplot as plt


class ImageProcess():
    def __init__(self, imagePath, mode=None):
        self.image = 255 - cv2.cvtColor(cv2.imread(imagePath), cv2.COLOR_BGR2GRAY)  # type: np.array
        self.spitImage = []
        self.mode = mode
        self.width = self.image.shape[1]
        self.heigh = self.image.shape[0]

        if mode == 1:
            self.Spitmode_of_1()
        else:
            self.Spitmode_of_2_point = []
            self.Spitmode_of_2()

    def Spitmode_of_1(self):
        self.spitImage = []
        self.spitPoint = []
        tempImage = np.copy(self.image)
        tempImage[tempImage[:, :] < 100] = 0
        tempImage[tempImage[:, :] > 100] = 255
        image_x_mean = tempImage.mean(axis=0)

        # print(image_x_mean)
        # plt.scatter(range(len(image_x_mean)),image_x_mean)
        # plt.show()

        xlist = []
        change = True
        for index, x in enumerate(image_x_mean):
            if x > 0.01 and change:
                xlist.append(index)
                change = False
            elif x < 0.01 and (change is not True):
                xlist.append(index)
                change = True

        ylist = []
        for y in range(int(len(xlist) / 2)):
            image_y_mean = tempImage[:, xlist[2 * y]: xlist[2 * y + 1]].mean(axis=1)
            change = True
            for index, x in enumerate(image_y_mean):
                if x > 0.01 and change:
                    ylist.append(index)
                    change = False
                elif x < 0.01 and (change is not True):
                    ylist.append(index)
                    change = True

        if len(ylist) != len(xlist):
            print("xlist: ", xlist)
            print("ylist: ", ylist)
            return None
        else:
            for i in range(int(len(xlist) / 2)):
                if ylist[2 * i] <= ylist[2 * i + 1]:
                    self.spitPoint.append([xlist[2 * i], ylist[2 * i], xlist[2 * i + 1], ylist[2 * i + 1]])
                else:
                    self.spitPoint.append([xlist[2 * i], ylist[2 * i + 1], xlist[2 * i + 1], ylist[2 * i]])

        self.spitImage = np.zeros([len(self.spitPoint), 28, 28])

        for index, (x, y, x1, y1) in enumerate(self.spitPoint):
            hight = y1 - y
            width = x1 - x
            x_WidenRate = max((hight / width) * 1.3, 1.5)
            y_WidenRate = max((width / hight) * 2, 1.5)
            T_image = np.zeros([int(hight * y_WidenRate), int(width * x_WidenRate)])
            y_start = int(hight * ((y_WidenRate - 1) / 2))
            x_start = int(width * ((x_WidenRate - 1) / 2))

            T_image[y_start: y_start + hight, x_start:x_start + width] = self.image[y:y1, x:x1]
            # 图片resize到28*28
            O_image = cv2.resize(T_image, (28, 28), interpolation=cv2.INTER_CUBIC)

            self.spitImage[index] = O_image

    def Spitmode_of_2(self):
        self.spitImage = []
        self.spitPoint = []
        tempImage = np.copy(self.image)
        tempImage[tempImage[:, :] < 100] = 0
        tempImage[tempImage[:, :] > 100] = 255

        for w in range(self.width):
            for h in range(self.heigh):
                if tempImage[h, w] == 255:
                    if self.mode == 2:
                        self.bfs(tempImage, w, h)
                    else:
                        self.dfs(tempImage, w, h)

                    self.Spitmode_of_2_point.append([-1, -1])

        numCount = 0
        for point in self.Spitmode_of_2_point:
            if point == [-1, -1]:
                numCount = numCount + 1
        log.info_out(str("get ") + str(numCount) + str(" image"))

        numpoint = []
        self.spitImage = np.zeros([numCount, 28, 28])
        numCount = 0
        for point in self.Spitmode_of_2_point:
            if point == [-1, -1]:
                np_numpoint = np.array(numpoint)
                x_max = max(np_numpoint[:, 0])
                x_min = min(np_numpoint[:, 0])
                y_max = max(np_numpoint[:, 1])
                y_min = min(np_numpoint[:, 1])
                # print(x_max, x_min, y_min, y_max)
                # print([y_max - y_min, x_max - x_min])
                hight = y_max - y_min
                width = x_max - x_min
                x_WidenRate = max((hight / width) * 1.3, 1.5)
                y_WidenRate = max((width / hight) * 2, 1.5)
                outImageTemp = np.zeros([int(hight * y_WidenRate), int(width * x_WidenRate)])
                y_start = int(hight * ((y_WidenRate - 1) / 2))
                x_start = int(width * ((x_WidenRate - 1) / 2))

                # outImageTemp = np.zeros([y_max - y_min + 1, x_max - x_min + 1])
                for x, y in numpoint:
                    outImageTemp[y - y_min + y_start, x - x_min + x_start] = self.image[y, x]

                T_image = cv2.resize(outImageTemp, (28, 28), interpolation=cv2.INTER_CUBIC)
                self.spitImage[numCount] = T_image
                self.spitPoint.append([x_min, y_min, x_max, y_max])
                numCount = numCount + 1
                numpoint = []
                continue

            numpoint.append(point)

    def dfs(self, image, x, y):
        if x > self.width or y > self.heigh or x < 0 or y < 0:
            return
        if image[y, x] != 255:
            return

        self.Spitmode_of_2_point.append([x, y])
        image[y, x] = 0

        for index_x in [-1, 0, 1]:
            for index_y in [-1, 0, 1]:
                self.dfs(image, x + index_x, y + index_y)

    def bfs(self, image, x, y):
        queue_point = []

        queue_point.append([y, x])
        while queue_point != []:
            y, x = queue_point.pop()
            self.Spitmode_of_2_point.append([x, y])
            image[y, x] = 0
            for y_index in [-1, 0, 1]:
                for x_index in [-1, 0, 1]:
                    if (y + y_index) >= 0 and (y + y_index) <= self.heigh and (x + x_index) >= 0 and (
                                x + x_index) <= self.width:
                        if image[y + y_index, x + x_index] == 255:
                            queue_point.append([y + y_index, x + x_index])

    def get_rect_point(self):
        return self.spitPoint

    def __getitem__(self, item):
        """
        :param item: 第几个图片
        :return: 28*28的图片
        """
        return self.spitImage[item]

    def __len__(self):
        if self.spitImage == []:
            return 0
        else:
            return self.spitImage.shape[0]
