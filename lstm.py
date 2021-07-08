
import tensorflow as tf

class Turtlebot_LSTM(tf.keras.Model):
    def __init__(self):
        super(Turtlebot_LSTM,self).__init__()
        self.Od=tf.keras.layers.LSTM(32)
        self.Sc=tf.keras.layers.LSTM(32)
        self.concat=tf.keras.layers.Concatenate()
        self.dense1=tf.keras.layers.Dense(64,activation=tf.nn.relu)
        self.dense2=tf.keras.layers.Dense(128,activation=tf.nn.relu)
        self.dense3=tf.keras.layers.Dense(3,activation=tf.nn.softmax)
    def call(self,inputs):
        x1,x2=inputs
        x1=self.Od(x1)
        x2=self.Sc(x2)
        x=self.concat([x1,x2])
        x=self.dense1(x)
        x=self.dense2(x)
        return(self.dense3(x))