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
import time

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

model_path="models/"
class ThresholdCallback(tf.keras.callbacks.Callback):
    def __init__(self, threshold):
        super(ThresholdCallback, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None): 
        val_acc = logs["val_categorical_accuracy"]
        acc = logs["categorical_accuracy"]
        if val_acc >= self.threshold and acc>=self.threshold :
            self.model.stop_training = True


def trainings(model,batch_sizes,num_epochs,ds,l,on_top=True):
    histories=[]
    test_scores=[]
    times=[]
    name=model().name
    if on_top:
        model_inst=model()
    for batch_size in batch_sizes:
        if not on_top:
            model_inst=model()
        print('Training {} Using batch size {}'.format(name,batch_size))
        train,validation,test=split_data(ds,l,batch_train=batch_size)
        cb=[tf.keras.callbacks.ModelCheckpoint(os.path.join(model_path,'{}_model/checkpoint_batch{}_epochs{}.h5'.format(name,batch_size,num_epochs)),monitor='val_loss',save_weight_only=True,save_best_only=True)
        #,ThresholdCallback(1)
        ,tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=100)
        ]
        model_inst.compile(optimizer='adam',loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
        start=time.time()
        histories.append(model_inst.fit(train,epochs=num_epochs,callbacks=cb,verbose=1,validation_data=validation).history)
        model_inst.load_weights(os.path.join(model_path,'{}_model/checkpoint_batch{}_epochs{}.h5'.format(name,batch_size,num_epochs)))
        test_scores.append(model_inst.evaluate(test))
        times.append(time.time() - start)
        model_inst.save(os.path.join(model_path,'{}_model/{}.h5'.format(name,batch_size)))
    return(histories,test_scores,times)

batch_sizes=[1,10,20,30]
num_epochs=200
results={}
ds,l=make_dataset(path='FullData/')
models=[Turtlebot_CNN]
#,Inception]#,Turtlebot_LSTM]#Models fed to the function are not instancied
for mod in models:
    name=mod().name
    if name=='lstm':
        with tf.device('/CPU:0'):#CPU is more efficient for LSTM architecture
            res=trainings(mod,batch_sizes,num_epochs,ds,l,on_top=False)          
    else :
        res=trainings(mod,batch_sizes,num_epochs,ds,l,on_top=False)
    results={'History': res[0],'Scores':res[1],'Times':res[2],'Batch sizes':batch_sizes}

    if os.path.exists("new_results_{}.json".format(name)):
        os.remove("new_results_{}.json".format(name))
    with open("new_results_{}.json".format(name),'w') as outfile:
        json.dump(results,outfile)

num_epochs=500
for mod in models:
    name=mod().name
    if name=='lstm':
        with tf.device('/CPU:0'):#CPU is more efficient for LSTM architecture
            res=trainings(mod,batch_sizes,num_epochs,ds,l,on_top=True)          
    else :
        res=trainings(mod,batch_sizes,num_epochs,ds,l,on_top=True)
    results={'History': res[0],'Scores':res[1],'Times':res[2],'Batch sizes':batch_sizes}

    if os.path.exists("results_{}.json".format(name)):
        os.remove("results_{}.json".format(name))
    with open("results_{}.json".format(name),'w') as outfile:
        json.dump(results,outfile)
# %%

