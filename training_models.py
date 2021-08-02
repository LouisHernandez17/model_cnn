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

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

def trainings(model,batch_sizes,num_epochs,ds,l,on_top=True):
    histories=[]
    test_scores=[]
    if on_top:
        model=model()
    for batch_size in batch_sizes:
        if not on_top:
            model=model()
        print('Training {} Using batch size {}'.format(model.short_name,batch_size))
        train,validation,test=split_data(ds,l,batch_train=batch_size)
        cb=[tf.keras.callbacks.ModelCheckpoint('{}_model/checkpoint_batch{}_epochs{}.h5'.format(model.short_name,batch_size,num_epochs),monitor='val_loss',save_weight_only=True,save_best_only=True),tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=50)]
        model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
        histories.append(model.fit(train,epochs=num_epochs,callbacks=cb,verbose=1,validation_data=validation).history)
        model.load_weights('{}_model/checkpoint_batch{}_epochs{}.h5'.format(model.short_name,batch_size,num_epochs))
        test_scores.append(model.evaluate(test))
    return(histories,test_scores)

batch_sizes=[1,10,20,30]
num_epochs=1500
results={}
ds,l=make_dataset(path='FullData/')
models=[Inception,Turtlebot_LSTM,Turtlebot_CNN]#Models fed to the function are not instancied
for mod in models:
    name=mod().short_name
    if name=='lstm':
        with tf.device('/CPU:0'):#CPU is more efficient for LSTM architecture
            res=trainings(mod,batch_sizes,num_epochs,ds,l)
    else :
        res=trainings(mod,batch_sizes,num_epochs,ds,l)
    results={'History': res[0],'Scores':res[1],'Batch sizes':batch_sizes}

    if os.path.exists("results_{}.json".format(name)):
        os.remove("results_{}.json".format(name))
    with open("results_{}.json".format(name),'w') as outfile:
        json.dump(results,outfile)
# %%