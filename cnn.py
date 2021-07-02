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
        if type=="cnn":
            self.Od1=tf.keras.layers.Conv1D(filters=64,kernel_size=10)
            self.Od2=tf.keras.layers.Conv1D(filters=32,kernel_size=10)
            self.Sc1=tf.keras.layers.Conv1D(filters=128,kernel_size=10)
            self.Sc2=tf.keras.layers.Conv1D(filters=64,kernel_size=10)
            self.Sc3=tf.keras.layers.Conv1D(filters=32,kernel_size=10)
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
        if self.type=='cnn':
            x2=self.avg_pool_sc(x2)
            x1=self.avg_pool_od(x1)
        x=self.concat([x1,x2])
        x=self.dense1(x)
        x=self.dense2(x)
        return(self.dense3(x))
    
# %%
def data_preparation(path):
    odoms,scans,labels=make_dataset(path)
    l=len(odoms)
    odoms_ds=tf.data.Dataset.from_generator(lambda:odoms,tf.float64,output_shapes=(None,13))
    scans_ds=tf.data.Dataset.from_generator(lambda:scans,tf.float64,output_shapes=(None,360))
    labels_ds=tf.data.Dataset.from_generator(lambda:labels,tf.float64)
    X=tf.data.Dataset.zip((odoms_ds,scans_ds))
    ds=tf.data.Dataset.zip((X,labels_ds)).batch(1)
    return ds,l
    

# %%
