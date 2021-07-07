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
    df=pd.DataFrame()
    num=int(os.path.basename(path))
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
    df_od=df_od[od_cols].fillna(method="ffill").fillna(value=0)
    sc_cols=[i for i in df_sc.columns if i.split('_')[0]=='intensities']
    df_sc=df_sc[sc_cols].fillna(method="ffill").fillna(value=0)
    return(df_od,df_sc)
    
# %%
def make_dataset(path):
    odoms=[]
    scans=[]
    labels=[]
    l=0
    for folder_class in os.listdir(path):
        if folder_class=='NoNoise':
            label=[1,0,0]
        elif folder_class=='OdomNoise':
            label=[0,1,0]
        elif folder_class=='ScanNoise':
            label=[0,0,1]
        for folder_data in os.listdir(os.path.join(path,folder_class)):
            df_od,df_sc=read_bag(os.path.join(path,folder_class,folder_data))
            if len(df_od)>15 and len(df_sc)>15:
                l+=1
                odoms.append(df_od.values.tolist())
                scans.append(df_sc.values.tolist())
                labels.append(label)
    odoms=tf.ragged.constant(odoms,ragged_rank=1)
    scans=tf.ragged.constant(scans,ragged_rank=1)
    odoms_ds=tf.data.Dataset.from_tensor_slices(odoms)
    scans_ds=tf.data.Dataset.from_tensor_slices(scans)
    labels_ds=tf.data.Dataset.from_tensor_slices(labels)
    X=tf.data.Dataset.zip((odoms_ds,scans_ds))
    ds=tf.data.Dataset.zip((X,labels_ds))
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