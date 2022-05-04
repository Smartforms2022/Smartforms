import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Dense, MaxPool2D, Flatten, BatchNormalization, Dropout, Lambda
from tensorflow.keras.regularizers import l2
class Model1():
    def __init__(self, input_size, num_labels):
        self.input_size = input_size
        self.num_labels = num_labels
                
    def get_model(self, freeze_backbone = False):
        h,w,c = self.input_size #30,30,1
        train_the_model = not freeze_backbone
        weight_decay1 = 0
        weight_decay2= weight_decay1
        model = Sequential([
        #Convolutional layer1
        Conv2D(filters=8, kernel_size=3, activation='relu', name='conv1', trainable=train_the_model, padding='same', input_shape=self.input_size, kernel_regularizer=l2(weight_decay1)),
        
        #Convolutional layer 2
        Conv2D(filters=16, kernel_size=3, activation='relu', name='conv2', trainable=train_the_model, padding='same',kernel_regularizer=l2(weight_decay1)),
        
        #Convolutional layer 3
        Conv2D(filters=32, kernel_size=3, activation='relu', name='conv3', trainable=train_the_model, padding='same',kernel_regularizer=l2(weight_decay2)),
        MaxPool2D(),
        
        #Convolutional layer 4
        Conv2D(filters=64, kernel_size=3, activation='relu', name='conv4', trainable=train_the_model, padding='same',kernel_regularizer=l2(weight_decay2)),
        MaxPool2D(),
        
        #Convolutional layer 5
        Conv2D(filters=128, kernel_size=3, activation='relu', name='conv5', trainable=train_the_model,kernel_regularizer=l2(weight_decay2), padding='same'),
        MaxPool2D(),
        
        #fully connected layer valid
        Flatten(),
        Dense(128, activation=None, name = 'fc1', trainable=False),
        Dropout(rate=0.5),
        Dense(10, activation='softmax', name = 'fc_out')
        ])
               
        return model
    
    def get_backbone(self, freeze_backbone=False):
        h,w,c = self.input_size #30,30,1
        train_the_model = not freeze_backbone
        weight_decay1=0#2e-6
        weight_decay2=0#2e-6
        model = Sequential([
        #Convolutional layer1
        Conv2D(filters=8, kernel_size=3, activation='relu', name='conv1', trainable=train_the_model,kernel_regularizer=l2(weight_decay1), padding='same', input_shape=self.input_size),
        #Convolutional layer 2
        Conv2D(filters=16, kernel_size=3, activation='relu', name='conv2', trainable=train_the_model,kernel_regularizer=l2(weight_decay1), padding='same'),
        
        #Convolutional layer 3
        Conv2D(filters=32, kernel_size=3, activation='relu', name='conv3', trainable=train_the_model,kernel_regularizer=l2(weight_decay2), padding='same'),
        MaxPool2D(),
        
        #Convolutional layer 4
        Conv2D(filters=64, kernel_size=3, activation='relu', name='conv4', trainable=train_the_model,kernel_regularizer=l2(weight_decay2), padding='same'),
        MaxPool2D(),
        
        #Convolutional layer 5
        Conv2D(filters=128, kernel_size=3, activation='relu', name='conv5', trainable=train_the_model,kernel_regularizer=l2(weight_decay2), padding='same'),
        MaxPool2D(),
        
        #fully connected layer
        Flatten(),
        Dense(128, name='fc1', activation=None), # No activation on final dense layer
        Dropout(rate=0.5),
        ])
        
        return model
        
