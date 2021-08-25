#%%
from cnn import Turtlebot_CNN
from lstm import Turtlebot_LSTM
from read_data import make_dataset,split_data
import tensorflow as tf
from inception import Inception
import numpy as np
import os

def predict(path_model,data):#Data must be a list of two arrays : the first one, with dimensions (Time,13) for the first one and (Time,360) for the second one.
    model=tf.keras.models.load_model(path_model)
    labels=['NoNoise','OdomNoise','ScanNoise']
    ragged_data=[]
    ragged_data.append(tf.ragged.constant(data[0]))
    ragged_data.append(tf.ragged.constant(data[1]))
    pred=model.predict(ragged_data)
    return labels[np.argmax(pred)]

data=[np.random.random((1,612,13)),np.random.random((1,358,360))]
for root,folders,files in os.walk('models'):
    for file in files:
        path_model=os.path.join(root,file)
        print(root)
        print(file)
        print(predict(path_model,data))
        print('-----------')
