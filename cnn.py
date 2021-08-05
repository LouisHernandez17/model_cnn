#%%
import tensorflow as tf
# %%
class Turtlebot_CNN(tf.keras.Model):
    def __init__(self,n_branches=2):
        super(Turtlebot_CNN,self).__init__()
        self.n_branches=n_branches
        self.short_name='cnn'
        self.convs1=[tf.keras.layers.Conv1D(filters=64,kernel_size=10,padding="same",use_bias=False) for i in range(self.n_branches)]
        self.convs2=[tf.keras.layers.Conv1D(filters=32,kernel_size=10,padding="same",use_bias=False) for i in range(self.n_branches)]
        self.avg_pools=[tf.keras.layers.GlobalAveragePooling1D() for i in range(self.n_branches)]
        self.concat=tf.keras.layers.Concatenate()
        self.dense1=tf.keras.layers.Dense(64,activation=tf.nn.relu)
        self.dense2=tf.keras.layers.Dense(128,activation=tf.nn.relu)
        self.dense3=tf.keras.layers.Dense(3,activation=tf.nn.softmax)
    def call(self,inputs):
        xs=[]
        for i,x in enumerate(inputs):
            x=x.to_tensor()
            x=self.convs1[i](x)
            x=self.convs2[i](x)
            x=self.avg_pools[i](x)
            xs.append(x)
        x=self.concat(xs)
        x=self.dense1(x)
        x=self.dense2(x)
        return(self.dense3(x))

    
# %%

    

# %%