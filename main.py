# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 11:20:27 2021

@author: 34296
"""

import cv2
import dlib
import PIL.Image as I
import torch
from torchvision import transforms
from cnn_for_fd_eye import *
from cnn_for_fd_mouth import *
from enhancement import *
import numpy as np
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QFileDialog, QMainWindow
from mainwindow import Ui_MainWindow

#导入模型
model_eye=torch.load('D:/学习相关/大三下/cv/fd/cnn_for_eye_16.pkl').cuda()
model_mouth=torch.load('D:/学习相关/大三下/cv/fd/cnn_for_mouth_16.pkl').cuda()

#阈值
thre=6 
#转pil为tensor
transform1=transforms.Compose([transforms.Grayscale(),transforms.Resize((40,40)),transforms.ToTensor()])
transform2=transforms.Compose([transforms.Grayscale(),transforms.Resize(40),transforms.RandomCrop(30),transforms.ToTensor(),transforms.Normalize([0.5],[0.5])])
font = cv2.FONT_HERSHEY_SIMPLEX
dict1=['open eye','closed eye']
dict2=['closed mouth','open mouth']

#使用dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    "shape_predictor_68_face_landmarks.dat"
) 

cap=cv2.VideoCapture(0, cv2.CAP_DSHOW)

#一段时间内的结果
t=0
seq=np.zeros((12,2))

#窗口
class PyQtMainEntry(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        timer = QtCore.QTimer(self)
        self.pos = QPixmap(r'p.jpg')
        self.neg = QPixmap(r'n.png')
        timer.start(100) #计时器，时间是毫秒
        timer.timeout.connect(self._queryFrame)

    @QtCore.pyqtSlot()
    def _queryFrame(self):
            
        '''
        循环捕获图片
        '''
        frame,flag = gainresult()
        if flag:
            img_rows, img_cols, channels = frame.shape
            bytesPerLine = channels * img_cols
    
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, frame)
            QImg = QImage(frame.data, img_cols, img_rows, bytesPerLine, QImage.Format_RGB888)
            self.label.setPixmap(QPixmap.fromImage(QImg).scaled(self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.deterfd()
    def deterfd(self):
        '''根据一段时间内闭眼张嘴频率计算判断疲劳驾驶与否'''
        global seq
        global thre
        eyefre=seq[:,0].sum()
        mouthfre=seq[:,1].sum()
        result=0.9*eyefre+0.7*mouthfre #quan
        if result>thre:
            self.label_2.setPixmap(self.neg)
        else:
            self.label_2.setPixmap(self.pos)    
    def closeEvent(self, event):
        cap.release()
        
def gainresult():
    global t
    global seq
    if not cap.isOpened():
        return (None,0)
    ret,img=cap.read()
    if not ret:
        return (None,0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = enhancement(gray)#图像增强
    dets = detector(gray, 1)
    if not dets:
        print('no face detected!')
        return (img,1)
    for face in dets:
        shape = predictor(img, face)  # 寻找人脸的68个标定点
        shape=shape.parts()
        lefteye=[shape[36].x,min(shape[37].y,shape[38].y),
                 shape[39].x,max(shape[41].y,shape[40].y)]
        righteye=[shape[42].x,min(shape[43].y,shape[44].y),
                 shape[45].x,max(shape[47].y,shape[46].y)]
        mouth=[shape[48].x,min(shape[50].y,shape[52].y),
                 shape[54].x,shape[57].y]
    
    dst1=gray[lefteye[1]-10:lefteye[3]+10,lefteye[0]-10:lefteye[2]+10]
    dst2=gray[righteye[1]-10:righteye[3]+10,righteye[0]-10:righteye[2]+10]
    dst3=gray[mouth[1]-10:mouth[3]+10,mouth[0]-10:mouth[2]+10]
    y1=model_eye(transform1(I.fromarray(dst1)).view(1,1,40,40).cuda())
    y2=model_eye(transform1(I.fromarray(dst2)).view(1,1,40,40).cuda())
    y3=model_mouth(transform2(I.fromarray(dst3)).view(1,1,30,30).cuda())
    
    
    _,y1_pred=torch.max(torch.softmax(y1.data,dim=1),dim=1)
    _,y2_pred=torch.max(torch.softmax(y2.data,dim=1),dim=1)
    _,y3_pred=torch.max(torch.softmax(y3.data,dim=1),dim=1)
    
    cv2.rectangle(img, (lefteye[0]-10,lefteye[1]-10), (lefteye[2]+10,lefteye[3]+10),(0, 0 , 255))
    cv2.rectangle(img, (righteye[0]-10,righteye[1]-10), (righteye[2]+10,righteye[3]+10),(0, 0 , 255))
    cv2.rectangle(img, (mouth[0]-10,mouth[1]-10), (mouth[2]+10,mouth[3]+10),(0, 0 , 255))
    
    cv2.putText(img, dict1[y1_pred], (lefteye[0]-10,lefteye[1]-10), font, 1, (255, 255, 255), 2)
    cv2.putText(img, dict1[y2_pred], (righteye[0]-10,righteye[1]-10), font, 1, (255, 255, 255), 2)
    cv2.putText(img, dict2[y3_pred], (mouth[0]-10,mouth[1]-10), font, 1, (255, 255, 255), 2)
    #print((y1_pred.item() and y2_pred.item()),y3_pred.item())
    seq[t]=[y1_pred.item() and y2_pred.item(),y3_pred.item()]
    t=(t+1)%12
    return (img,1)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = PyQtMainEntry()
    window.show()
    sys.exit(app.exec_())
    




