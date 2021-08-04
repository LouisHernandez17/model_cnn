
import tensorflow as tf

class Turtlebot_LSTM(tf.keras.Model):
    def __init__(self,n_branches=2):
        super(Turtlebot_LSTM,self).__init__()
        self.n_branches=n_branches
        self.lstms=[tf.keras.layers.LSTM(32) for i in range(self.n_branches)]
        self.concat=tf.keras.layers.Concatenate()
        self.dense1=tf.keras.layers.Dense(64,activation=tf.nn.relu)
        self.dense2=tf.keras.layers.Dense(128,activation=tf.nn.relu)
        self.dense3=tf.keras.layers.Dense(3,activation=tf.nn.softmax)
        self.short_name='lstm'
    def call(self,inputs):
        xs=[]
        for i,x in enumerate(inputs):
            xs.append(self.lstms[i](x))
        x=self.concat(xs)
        x=self.dense1(x)
        x=self.dense2(x)
        return(self.dense3(x))