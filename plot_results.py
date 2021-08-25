#%%

import matplotlib.pyplot as plt
import json
import seaborn as sns
import pandas as pd
import time
names=['cnn','lstm','inc']
paths=['results_{}.json'.format(name) for name in names]

colors=['green','blue','red','magenta']
for i,path in enumerate(paths):
    with open(path) as f:
        data=json.load(f)
    x1=0
    x2=0
    y=[]
    plt.figure()
    plt.title('Learning curve for {} model'.format(names[i]))
    plt.ylim([-0.05,1.05])
    for j,history in enumerate(data['History']):
        x2+=len(history['loss'])
        y+=history['loss']
        plt.plot(list(range(x1,x2)),history['loss'],color=colors[j],label='Batch size {}'.format(data['Batch sizes'][j]))
        plt.plot(list(range(x1,x2)),history['val_loss'],color='dark'+colors[j])
        plt.xlabel('Epochs')
        plt.ylabel('Category Crossentropy')
        x1=x2

    plt.legend()
    plt.savefig('{}_train.eps'.format(names[i]))
    plt.show()
    plt.figure()
    plt.plot(data['Batch sizes'],data['Scores'])
    plt.legend(('Loss','Accuracy'))
    plt.xlabel('Batch size')
    plt.title('Score on test dataset for each batch size')
    plt.savefig('results_{}.eps'.format(names[i]))
    plt.show()
    print(names[i])
    print(data['Scores'])

new_paths=['new_results_{}.json'.format(name) for name in names]
df=pd.DataFrame(columns=['model','time','batch size','accuracy'])
for i,path in enumerate(new_paths):
    with open(path) as f:
        data=json.load(f)
    for j,batch_size in enumerate(data['Batch sizes']):
        df.loc[names[i]+'_'+str(batch_size)]=[names[i],data['Times'][j],batch_size,data['Scores'][j][1]]
plt.figure()
g=sns.relplot(data=df,x='accuracy',y='time',size='batch size',hue='model')
plt.plot([1,1],[0,1500],'k--')
plt.savefig('results.eps')
plt.show()
# %%

# %%
