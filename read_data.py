#%%
import numpy as np
import os
import pandas as pd
import bagpy
import tensorflow as tf
import math
from tensorflow.python.ops.functional_ops import scan
#%%
def read_bag(path):
    for f in os.listdir(path):
        if f[-4:]=='.bag':
            bag=bagpy.bagreader(os.path.join(path,f))
    odom=bag.odometry_data()
    scan=bag.message_by_topic("/scan")
    #Opens as Dataframes
    df_od=pd.read_csv(odom[0],index_col='Time')
    df_sc=pd.read_csv(scan,index_col='Time')
    #Removes useless columns
    od_cols=[i for i in df_od.columns if i.split('.')[0]=='pose' or i.split('.')[0]=='orientation' or i.split('.')[0]=='linear' or i.split('.')[0]=='angular']
    df_od=df_od[od_cols].fillna(method="ffill").fillna(value=0)#Fills the NaNs
    sc_cols=[i for i in df_sc.columns if i.split('_')[0]=='intensities']
    df_sc=df_sc[sc_cols].fillna(method="ffill").fillna(value=0)
    return(df_od,df_sc)
    
# %%
def make_dataset(path,with_label=True):
    odoms=[]
    scans=[]
    labels_name=['NoNoise','OdomNoise','ScanNoise']
    labels=[]
    l=0
    if with_label:#We read several folders with the foldername indicating the label
        for i,lab_name in enumerate(labels_name):
            for folder_class in os.listdir(path):
                label=[0 for i in range(len(labels_name))]
                if folder_class==lab_name:
                    label[i]=1
                    for folder_data in os.listdir(os.path.join(path,folder_class)):
                        df_od,df_sc=read_bag(os.path.join(path,folder_class,folder_data))
                        if len(df_od)>15 and len(df_sc)>15:
                            l+=1
                            odoms.append(df_od.values.tolist())
                            scans.append(df_sc.values.tolist())
                            labels.append(label)
                else:
                    ()
        odoms=tf.ragged.constant(odoms,ragged_rank=1)
        scans=tf.ragged.constant(scans,ragged_rank=1)
        odoms_ds=tf.data.Dataset.from_tensor_slices(odoms)
        scans_ds=tf.data.Dataset.from_tensor_slices(scans)
        labels_ds=tf.data.Dataset.from_tensor_slices(labels)
        X=tf.data.Dataset.zip((odoms_ds,scans_ds))
        ds=tf.data.Dataset.zip((X,labels_ds))
        return ds,l
    else:
        if os.path.basename(path).isnumeric():
            df_od,df_sc=read_bag(path)
            if len(df_od)>15 and len(df_sc)>15:
                l+=1
                odoms.append(df_od.values.tolist())
                scans.append(df_sc.values.tolist())
                labels.append([0,0,0])
            odoms=tf.ragged.constant(odoms,ragged_rank=1)
            scans=tf.ragged.constant(scans,ragged_rank=1)
            odoms_ds=tf.data.Dataset.from_tensor_slices(odoms)
            scans_ds=tf.data.Dataset.from_tensor_slices(scans)
            labels_ds=tf.data.Dataset.from_tensor_slices(labels)
            X=tf.data.Dataset.zip((odoms_ds,scans_ds))
            ds=tf.data.Dataset.zip((X,labels_ds)).batch(1)
            return ds,l
        else:
            for folder_data in os.listdir(path):
                df_od,df_sc=read_bag(os.path.join(path,folder_data))
                if len(df_od)>15 and len(df_sc)>15:
                    l+=1
                    odoms.append(df_od.values.tolist())
                    scans.append(df_sc.values.tolist())
                    labels.append([0,0,0])
            odoms=tf.ragged.constant(odoms,ragged_rank=1)
            scans=tf.ragged.constant(scans,ragged_rank=1)
            odoms_ds=tf.data.Dataset.from_tensor_slices(odoms)
            scans_ds=tf.data.Dataset.from_tensor_slices(scans)
            labels_ds=tf.data.Dataset.from_tensor_slices(labels)
            X=tf.data.Dataset.zip((odoms_ds,scans_ds))
            ds=tf.data.Dataset.zip((X,labels_ds)).batch(1)
            return ds,l


# %%
def split_data(ds,l,batch_train=10,prop_train_val=0.8,prop_train=0.8):
    ds=ds.shuffle(l)
    l_train_val=math.ceil(prop_train_val*l)
    train_val=ds.take(l_train_val)
    test=ds.skip(l_train_val)
    l_train=math.floor(prop_train*l_train_val)
    train=train_val.take(l_train)
    validation=train_val.skip(l_train)
    train=train.batch(batch_train)
    validation=validation.batch(1)
    test=test.batch(1)
    return(train,validation,train)
#%%
def pred2str(pred):
    labels_name=['NoNoise','OdomNoise','ScanNoise']
    pred=np.array(pred).reshape((3,))
    return labels_name[pred.argmax]
def predict(model,data):
    for i in data:
        print(pred2str(model.predict(data)))
