#%%
import tensorflow as tf
from read_data import make_dataset
import numpy as np
import math
import random
# %%
class Turtlebot_CNN(tf.keras.Model):
    def __init__(self,type="cnn"):
        super(Turtlebot_CNN,self).__init__()
        self.input_od=tf.keras.layers.InputLayer(input_shape=(None,None,9))
        self.input_sc=tf.keras.layers.InputLayer(input_shape=(None,None,360))
        self.type=type
        if type=="cnn":
            self.Od1=tf.keras.layers.Conv1D(filters=64,kernel_size=10,activation=tf.nn.relu)
            self.Od2=tf.keras.layers.Conv1D(filters=32,kernel_size=10,activation=tf.nn.relu)
            self.Sc1=tf.keras.layers.Conv1D(filters=128,kernel_size=25,activation=tf.nn.relu)
            self.Sc2=tf.keras.layers.Conv1D(filters=64,kernel_size=10,activation=tf.nn.relu)
            self.Sc3=tf.keras.layers.Conv1D(filters=32,kernel_size=10,activation=tf.nn.relu)
        elif type=="lstm":
            self.Od1=tf.keras.layers.LSTM(64,return_sequences=True)
            self.Od2=tf.keras.layers.LSTM(32)
            self.Sc1=tf.keras.layers.LSTM(128,return_sequences=True)
            self.Sc2=tf.keras.layers.LSTM(64,return_sequences=True)
            self.Sc3=tf.keras.layers.LSTM(32)

        self.avg_pool_od=tf.keras.layers.GlobalAveragePooling1D()
        self.avg_pool_sc=tf.keras.layers.GlobalAveragePooling1D()
        self.concat=tf.keras.layers.Concatenate()
        self.dense1=tf.keras.layers.Dense(64,activation=tf.nn.relu)
        self.dense2=tf.keras.layers.Dense(128,activation=tf.nn.relu)
        self.dense3=tf.keras.layers.Dense(3,activation=tf.nn.softmax)
    def call(self,inputs):
        x1=inputs[0]
        x2=inputs[1]
        x1=self.input_od(x1)
        x1=self.Od1(x1)
        x1=self.Od2(x1)
        x2=self.input_sc(x2)
        x2=self.Sc1(x2)
        x2=self.Sc2(x2)
        x2=self.Sc3(x2)
        if self.type=='cnn':
            x2=self.avg_pool_sc(x2)
            x1=self.avg_pool_od(x1)
        x=self.concat([x1,x2])
        x=self.dense1(x)
        x=self.dense2(x)
        return(self.dense3(x))
    
# %%
class Sequence_1by1(tf.keras.utils.Sequence):
    def __init__(self,x,y):
        self.x,self.y=x,y
        self.batch_size=1
    def __len__(self):
        return math.ceil(len(self.x[0]) / self.batch_size)

    def __getitem__(self, idx):
        batch_x1 = self.x[0][idx * self.batch_size:(idx + 1) *
        self.batch_size]
        batch_x2 = self.x[1][idx * self.batch_size:(idx + 1) *
        self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) *
        self.batch_size]

        return [np.array(batch_x1),np.array(batch_x2)], np.array(batch_y)


def data_preparation(path,training=0.7):
    odoms,scans,labels=make_dataset(path)
    l=list(zip(odoms,scans,labels))
    random.shuffle(l)
    i=math.ceil((len(l)-1)*training)
    train_list=l[:i]
    test_list=l[i:]
    odoms_train,scans_train,labels_train=zip(*train_list)
    odoms_test,scans_test,labels_test=zip(*test_list)
    train=Sequence_1by1([odoms_train,scans_train],labels_train)
    test=Sequence_1by1([odoms_test,scans_test],labels_test)
    return train,test
    

# %%