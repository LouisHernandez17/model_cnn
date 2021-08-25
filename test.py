#%%
from cnn import Turtlebot_CNN
from lstm import Turtlebot_LSTM
from read_data import make_dataset,split_data
import tensorflow as tf
import matplotlib.pyplot as plt
from inception import Inception
import os
import sys
import json
import pandas as pd
import numpy as np
import time
ds,l=make_dataset(path='FullData/')
#%%
model_path="/media/louis/TOSHIBA EXT/models"
for (a,b),c in ds:
    inps=(a,b)
model=tf.keras.models.load_model(os.path.join(model_path,'inc_model/1'))
model.predict(inps)
# %%
