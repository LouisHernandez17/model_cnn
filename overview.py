#%%
from cnn import Turtlebot_CNN
from lstm import Turtlebot_LSTM
from read_data import make_dataset,split_data
import tensorflow as tf
import matplotlib.pyplot as plt
from inception import Inception
import pandas as pd
import os
results=pd.DataFrame(columns=['Batch size','Epochs','Loss','Accuracy'])
ds,l=make_dataset(path='FullData/')
models=[Inception(),Turtlebot_LSTM(),Turtlebot_CNN()]
train,validation,test=split_data(ds,l)
#%%
for model in models:
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
    model.fit(train,epochs=1,validation_data=validation)
    for root,dirs,files in os.walk('{}_model/'.format(model.short_name)):
        for file in files:
            batch,epochs=file.split('_')[5:],file.split('_')[2][6:-2]
            model.load_weights(file)
            loss,accuracy=model.evaluate(ds.batch(1))
        results.loc[file]=[model.short_name,batch,epochs,loss,accuracy]
results.to_csv('Results.csv')
# %%
