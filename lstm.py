#%%
import tensorflow as tf
import numpy as np


def LSTM_branch(input,dim):
    x=tf.keras.layers.LSTM(32)(input)
    return x

def Turtlebot_LSTM(dims=[13,360]):
    outs=[]
    Inputs=[tf.keras.Input(shape=(None,dim),ragged=True) for dim in dims]
    for i,dim in enumerate(dims):
        outs.append(LSTM_branch(Inputs[i],dim))
    concat=tf.keras.layers.Concatenate()(outs)
    dense1=tf.keras.layers.Dense(64,activation=tf.nn.relu)(concat)
    dense2=tf.keras.layers.Dense(3,activation=tf.nn.softmax)(dense1)
    return(tf.keras.Model(Inputs,dense2,name='lstm'))