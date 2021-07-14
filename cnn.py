#%%
import tensorflow as tf
# %%
class Turtlebot_CNN(tf.keras.Model):
    def __init__(self):
        super(Turtlebot_CNN,self).__init__()
        self.mask_od=tf.keras.layers.Masking(mask_value=0)
        self.mask_sc=tf.keras.layers.Masking(mask_value=0)
        self.Od1=tf.keras.layers.Conv1D(filters=64,kernel_size=10,padding="same",use_bias=False)
        self.Od2=tf.keras.layers.Conv1D(filters=32,kernel_size=10,padding="same",use_bias=False)
        self.Sc1=tf.keras.layers.Conv1D(filters=128,kernel_size=10,padding="same",use_bias=False)
        self.Sc2=tf.keras.layers.Conv1D(filters=64,kernel_size=10,padding="same",use_bias=False)
        self.Sc3=tf.keras.layers.Conv1D(filters=32,kernel_size=10,padding="same",use_bias=False)
        self.avg_pool_od=tf.keras.layers.GlobalAveragePooling1D()
        self.avg_pool_sc=tf.keras.layers.GlobalAveragePooling1D()
        self.concat=tf.keras.layers.Concatenate()
        self.dense1=tf.keras.layers.Dense(64,activation=tf.nn.relu)
        self.dense2=tf.keras.layers.Dense(128,activation=tf.nn.relu)
        self.dense3=tf.keras.layers.Dense(3,activation=tf.nn.softmax)
    def call(self,inputs):
        x1,x2=inputs
        x1=self.mask_od(x1.to_tensor(default_value=0))
        x2=self.mask_sc(x2.to_tensor(default_value=0))
        x1=self.Od1(x1)
        x1=self.Od2(x1)
        x2=self.Sc1(x2)
        x2=self.Sc2(x2)
        x2=self.Sc3(x2)
        x2=self.avg_pool_sc(x2)
        x1=self.avg_pool_od(x1)
        x=self.concat([x1,x2])
        x=self.dense1(x)
        x=self.dense2(x)
        return(self.dense3(x))

    
# %%

    

# %%