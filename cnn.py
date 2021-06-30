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
        self.type=type
        self.input_Od=tf.keras.layers.InputLayer(input_shape=(None,None,13),batch_size=1)
        self.input_Sc=tf.keras.layers.InputLayer(input_shape=(None,None,360),batch_size=1)
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
        x1,x2=inputs
        x1=self.Od1(x1)
        x1=self.Od2(x1)
        x2=self.Sc1(x2)
        x2=self.Sc2(x2)
        x2=self.Sc3(x2)
        print(x2)
        if self.type=='cnn':
            x2=self.avg_pool_sc(x2)
            x1=self.avg_pool_od(x1)
        x=self.concat([x1,x2])
        x=self.dense1(x)
        x=self.dense2(x)
        return(self.dense3(x))
    
# %%
def data_preparation(path,training=0.7):
    odoms,scans,labels=make_dataset(path)
    i=math.ceil((len(odoms)-1)*training)
    odoms_train,scans_train,labels_train=odoms[:i],scans[:i],labels[:i]
    odoms_test,scans_test,labels_test=odoms[i:],scans[i:],labels[i:]
    odoms_train_ds=tf.data.Dataset.from_generator(lambda:odoms_train,tf.float64,output_shapes=(None,13))
    scans_train_ds=tf.data.Dataset.from_generator(lambda:scans_train,tf.float64,output_shapes=(None,360))
    labels_train_ds=tf.data.Dataset.from_generator(lambda:labels_train,tf.float64)
    odoms_test_ds=tf.data.Dataset.from_generator(lambda:odoms_test,tf.float64,output_shapes=(None,13))
    scans_test_ds=tf.data.Dataset.from_generator(lambda:scans_test,tf.float64,output_shapes=(None,630))
    labels_test_ds=tf.data.Dataset.from_generator(lambda:labels_test,tf.float64)
    X_train=tf.data.Dataset.zip((odoms_train_ds,scans_train_ds))
    X_test=tf.data.Dataset.zip((odoms_test_ds,scans_test_ds))
    train=tf.data.Dataset.zip((X_train,labels_train_ds)).batch(1)
    test=tf.data.Dataset.zip((X_test,labels_test_ds)).batch(1)
    return train,test
    

# %%
