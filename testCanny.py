import numpy as np
import pyfirmata
import time
import cv2
import pickle
import pandas as pd 
import os
import cv2
import matplotlib.pyplot as plt 
import seaborn as sns
import os
import serial
from PIL import Image
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import keras
from keras.layers import Dense, Conv2D
from keras.layers import Flatten
from keras.layers import MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.models import Sequential
from keras import backend as K
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from keras import optimizers
import pywt
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import pickle

# intialize the webcam and pass a constant which is 0
cam = cv2.VideoCapture(0)
# RELAY_PIN  = board.get_pin('a:5:o')

# title of the app
cv2.namedWindow('python webcam screenshot app')

board = pyfirmata.Arduino('COM5')

# let's assume the number of images gotten is 0
img_counter = 0
pickled_model = pickle.loads(open('hog_model2.pkl', 'rb'))

# pickled_model.predict(X_test)
data = []
# while loop
os.system('cls')
print('Selamat Datang')
print('1. Tekan Spasi untuk membuka pintu')
print('2. Tekan Esc untuk menutup aplikasi')
while True:
    # intializing the frame, ret
    ret, frame = cam.read()
    # if statement
    if not ret:
        print('failed to grab frame')
        break
    # the frame will show with the title of test
    cv2.imshow('test', frame)
    #to get continuous live video feed from my laptops webcam
    k  = cv2.waitKey(1)
    # if the escape key is been pressed, the app will stop
    if k%256 == 27:
        # print('escape hit, closing the app')
        break
    # if the spacebar key is been pressed
    # screenshots will be taken
    elif k%256  == 32:
        # the format for storing the images scrreenshotted
        img_name = 'test.jpg'
        # saves the image as a png file
        cv2.imwrite(img_name, frame)
        # print('screenshot taken')
        image = cv2.imread("test.jpg")
        
        image_array = Image.fromarray(image , 'RGB')
        resize_img = image_array.resize((50 , 50))
        
        resize_img = np.float32(resize_img) / 255.0
        
        # Calculate gradient 
        gx = cv2.Sobel(resize_img, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(resize_img, cv2.CV_32F, 0, 1, ksize=1)
        mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        
        # image_array = Image.fromarray(image , 'RGB')
        # resize_img = image_array.resize((50 , 50))
        # coeffs1 = pywt.dwt2(resize_img, 'bior1.3')
        # LL, (LH, HL, HH) = coeffs1
        data.append(np.array(mag))
        data.append(np.array(angle))
        data.append(np.array(gx))
        data.append(np.array(gy))
        cells = np.array(data, dtype=object)
        n = np.arange(cells.shape[0])
        np.random.shuffle(n)
        cells = cells[n]
        
        cells = cells.astype(np.float32)
        cells = cells/255

        x_test = cells

        predict = pickled_model.predict(x_test)

        # print(predict)
        yhat_classes=np.argmax(predict,axis=1)
        yhat_classes = to_categorical(yhat_classes, num_classes = 3)
        # status = ''.join([str(elem) for elem in yhat_classes[0]])
        # print(yhat_classes)
        if (yhat_classes[0][0] == 1) & (yhat_classes[0][1] == 0) & (yhat_classes[0][2] == 0):
            print('Penghuni Rumah')
            print("Communication Successfully started")
            board.digital[2].write(1)
            board.digital[3].write(1)
            
            # arduino = serial.Serial(port='COM5', baudrate=115200, timeout=.1)
            print("Pintu Terbuka")
            time.sleep(10)
            print("Pintu Tertutup")
            board.digital[2].write(0)
            board.digital[3].write(0)
            # value = write_read(num)
            # arduino.write(value.encode("utf-8"))
            # print (value)
            # print ("Data Terkirim ke Arduino") # mungkin belum sampai koneksi ke arduino nya
            time.sleep(5)
        else:
            print ("Bukan Penghuni")
        # the number of images automaticallly increases by 1
        img_counter += 1
        yhat_classes = np.delete(yhat_classes,0,0)
        yhat_classes = np.delete(yhat_classes,0,0)
        yhat_classes = np.delete(yhat_classes,0,0)
        yhat_classes = np.delete(yhat_classes,0,0)
        predict = np.delete(predict,0,0)
        predict = np.delete(predict,0,0)
        predict = np.delete(predict,0,0)
        predict = np.delete(predict,0,0)
        # print(predict)
        # print(yhat_classes)


# release the camera
cam.release()

# stops the camera window
cam.destoryAllWindows()