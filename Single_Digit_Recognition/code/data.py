import math
from os.path import join
import numpy as np
import pandas as pd
from PIL import Image,ImageOps
from tensorflow.keras.utils import Sequence
import cv2
import pickle
import random

class DigitDataset(Sequence):
    
    
    def __init__(self, img_list, img_type, batch_size, img_data_path, labels_path, random_crop=False, isClassifier=False):
        self.random_crop = random_crop
        self.img_list = img_list
        self.batch_size = batch_size
        self.same_form_list = []
        self.img_data_path = img_data_path
        self.labels = {}
        self.images = {}
        self.isClassifier = isClassifier
        self.img_type = img_type
        self.load_labels(labels_path)
        self.collect_imgs()
        self.digit_info_list()
        
    def __len__(self):
        return math.ceil(self.data_len / self.batch_size)

    def digit_info_list(self):
        self.digit_list = []
        for img_id in self.img_list:
            ph_no_list = self.labels[img_id]
            for i in range(len(ph_no_list)):
                ph_no = ph_no_list[i]
                for j in range(10):
                    if ph_no[j] != 'X':
                        self.digit_list.append([img_id,i,j])

        self.data_len = len(self.digit_list)
        print(self.data_len)
                   
    def preprocess(self,digit):
        digit = digit.reshape((30,30))
        #    digit[digit<200] = 0
        #    digit[digit>=200] = 255
          
        digit = digit.reshape((30,30,1))
        return (255-digit)/255
                    
    def __getitem__(self, index):
        bs = self.batch_size
        start = index*bs
        end = min(self.data_len, start+bs)

        #print(end)
        inputs = []
        labels = []

        for idx in range(start,end):
            x,y = self.get_form_data(idx)
            inputs.append(x)
            labels.append(y)

        inputs=np.array(inputs)
        labels=np.array(labels)
        
        if self.isClassifier == True:
            y = np.zeros((labels.shape[0],10))
            y[np.arange(labels.shape[0]), labels] = 1
            labels = y
            
        return inputs,labels

    def get_labels(self):
        bs = self.batch_size
        start = 0
        end = self.data_len
        inputs = []
        labels = []
        for idx in range(start,end):
            x,y = self.get_form_data(idx)
            inputs.append(x)
            labels.append(y)
        inputs=np.array(inputs)
        labels=np.array(labels)

        if self.isClassifier == True:
            y = np.zeros((labels.shape[0],10))
            y[np.arange(labels.shape[0]), labels] = 1
            labels = y

        return inputs,labels
 
    def load_img(self, idx):
        #print(self.img_type)
        path = join(self.img_data_path, str(idx)+'.'+self.img_type)
        img = Image.open(path)
        img = ImageOps.grayscale(img)
        img = np.array(img)
        img = np.array(img, dtype=np.uint8)

        return img
    
    def collect_imgs(self):  
        for idx in self.img_list:
            #print(idx)
            img = self.load_img(idx)
            img = self.remove_padding(img,idx)           
            img = np.expand_dims(img,-1)
            self.images[idx] = img
            
    def remove_padding(self,img,idx):
        new_img = np.zeros((16*30,10*30))
        n = (len(self.labels[idx]))
        for i in range(n):
            
            ph_num = self.labels[idx][i]
            for j in range(10):
                if ph_num[j] == 'X':
                    img[i*32:(i+1)*32,j*32:(j+1)*32] = 0
                else:
                    digit = img[i*32:(i+1)*32, j*32:(j+1)*32][1:-1, 1:-1]
                    new_img[i*30:(i+1)*30, j*30:(j+1)*30] = digit
                    
        return new_img
    
    
    def load_labels(self,labels_path):
        labels = pd.read_csv(labels_path, dtype='str')
        for form_id in self.img_list:
            labels_list = labels[labels["fileName"] == form_id]["phoneNumber"].tolist()
            self.labels[form_id] = labels_list

    def crop_digit(self,img,i,j):
        h,w,c = img.shape
        #print(type(i), i)
        x = i*30
        y = j*30  
        cx = x+15
        cy = y+15
        
        if self.random_crop:
            cx += np.random.randint(-5,6)
            cy += np.random.randint(-5,6)
        x1 = cx-15
        y1 = cy-15
        x1 = np.clip(x1,0,h-30)
        y1 = np.clip(y1,0,w-30) 
        x2 = x1+30
        y2 = y1+30
        
        return img[x1:x2, y1:y2]
    
    def get_digit(self,idx):
        #print(idx)
        digit_info = self.digit_list[idx]
        #print(digit_info)
        img = self.images[digit_info[0]]
        digit = self.crop_digit(img,digit_info[1], digit_info[2])
        return digit 
        
    def get_form_data(self, idx):
        digit = self.get_digit(idx)     
        digit = self.preprocess(digit)
        digit = digit.astype('float32')
        digit_info = self.digit_list[idx]
        number = self.labels[digit_info[0]][digit_info[1]]
        label_list = number#.split('_')       
        label = label_list[digit_info[2]]
        if label == 'a':
            y = 10
        else:
            y = int(label)
        return digit,y
