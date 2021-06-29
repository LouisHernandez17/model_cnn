#%%
import numpy as np
import os
import pandas as pd
import bagpy
#%%
def read_bag(path):
    df=pd.DataFrame()
    num=int(os.path.basename(path))
    for f in os.listdir(path):
        if f[-4:]=='.bag':
            bag=bagpy.bagreader(os.path.join(path,f))
        elif num<21:#No Noise
            label=[1,0,0]
        elif num<41:#Scan Noise
            label=[0,1,0]
        elif num<61:#Odom Noise
            label=[0,0,1]
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
    return(df_od,df_sc,label)
    
# %%
def make_dataset(path):
    odoms=[]
    scans=[]
    labels=[]
    for folder in os.listdir(path):
        df_od,df_sc,label=read_bag(os.path.join(path,folder))
        odoms.append(df_od.values)
        scans.append(df_sc.values)
        labels.append(label)
    return(odoms,scans,labels)
# %%