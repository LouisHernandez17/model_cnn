
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import tensorflow as tf
def CNN_branch(input):
    try :
        input=input.to_tensor()
    except:
        ()
    x=tf.keras.layers.Conv1D(filters=64,kernel_size=10,padding='same')(input)
    x=tf.keras.layers.Conv1D(filters=64,kernel_size=15,padding="same")(x)
    x=tf.keras.layers.Conv1D(filters=32,kernel_size=10,padding="same")(x)
    x=tf.keras.layers.Conv1D(filters=16,kernel_size=5,padding="same")(x)
    x=tf.keras.layers.GlobalMaxPool1D()(x)
    return x

def Turtlebot_CNN(dims=[13,360]):
    outs=[]
    Inputs=[tf.keras.Input(shape=(None,dim),ragged=True) for dim in dims]
    for i in range(len(dims)):
        outs.append(CNN_branch(Inputs[i]))
    concat=tf.keras.layers.Concatenate()(outs)
    dense1=tf.keras.layers.Dense(64,activation=tf.nn.relu)(concat)
    dense2=tf.keras.layers.Dense(3,activation=tf.nn.softmax)(dense1)
    return(tf.keras.Model(Inputs,dense2,name='cnn'))

# %%
