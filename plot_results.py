#%%
import matplotlib.pyplot as plt
import json
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
    for j,history in enumerate(data['History']):
        x2+=len(history['loss'])
        y+=history['loss']
        plt.plot(list(range(x1,x2)),history['loss'],color=colors[j],label='Training' if j==0 else '')
        plt.plot(list(range(x1,x2)),history['val_loss'],color='dark'+colors[j],label='Validation' if j==0 else '')
        x1=x2
    plt.legend()
    plt.show()
    plt.figure()
    plt.plot(data['Batch Sizes'],data['Scores'])
    plt.xlabel('Batch size')
    plt.title('Score on test dataset for each batch size')
# %%
