#%%
import os
import pandas as pd
from read_data import read_bag
import matplotlib.pyplot as plt

labels_name =["NoNoise",'OdomNoise','ScanNoise']
path="FullData"
colors=['r','g','b']
odoms=[]
scans=[]
labels=[]
for i,lab_name in enumerate(labels_name):
    for folder_class in os.listdir(path):
        label=[0 for i in range(len(labels_name))]
        if folder_class==lab_name:
            label[i]=1
            for folder_data in os.listdir(os.path.join(path,folder_class)):
                df_od,df_sc=read_bag(os.path.join(path,folder_class,folder_data))
                odoms.append(df_od)
                scans.append(df_sc)
                labels.append(i)
#%%
plt.figure()
for i,df_od in enumerate(odoms):
    plt.plot(df_od,color=colors[labels[i]],label=labels_name[labels[i]])
plt.show()
plt.figure()
for i,df_sc in enumerate(scans):
    print(df_sc)
    plt.plot(df_sc,color=colors[labels[i]],label=labels_name[labels[i]])
plt.show()
                    
# %% Scans are all =0 ??
