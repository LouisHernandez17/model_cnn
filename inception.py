import tensorflow as tf

#%%
def InceptionBlock(input,bottleneck_size=32,nb_filters=32):
    bottleneck=tf.keras.layers.Conv1D(filters=bottleneck_size,kernel_size=1,padding='same',use_bias=False)
    x=bottleneck(input)
    kernel_sizes=[40,20,10]
    outs=[]
    for kernel_size in kernel_sizes:
        conv=tf.keras.layers.Conv1D(filters=nb_filters,padding='same',kernel_size=kernel_size,use_bias=False)
        outs.append(conv(x))
    residual=tf.keras.layers.MaxPool1D(pool_size=3,padding='same',strides=1)(input)
    outs.append(tf.keras.layers.Conv1D(filters=nb_filters,padding='same',kernel_size=1,use_bias=False)(residual))
    out=tf.keras.layers.Concatenate()(outs)
    out=tf.keras.layers.Activation(activation='relu')(out)
    return(out)

def InceptionBranch(input,depth=6):
    try :
        input=input.to_tensor()
    except :
        ()
    x=input
    inp=input
    for i in range(depth):
        x=InceptionBlock(x)
        if i%3==2:
            new_inp=tf.keras.layers.Conv1D(filters=x.shape[-1],kernel_size=1,padding='same',use_bias=False)(inp)
            x=tf.keras.layers.Add()([x,new_inp])
            x=tf.keras.layers.Activation(activation='relu')(x)
            inp=x
    x=tf.keras.layers.GlobalAveragePooling1D()(x)
    return(x)
def Inception(dims=[13,360]):
    Inputs=[tf.keras.Input(shape=(None,dim),ragged=True) for dim in dims]
    outs=[]
    for i in range(len(dims)):
        outs.append(InceptionBranch(Inputs[i]))
    concat=tf.keras.layers.Concatenate()(outs)
    dense1=tf.keras.layers.Dense(64,activation=tf.nn.relu)(concat)
    dense2=tf.keras.layers.Dense(3,activation=tf.nn.softmax)(dense1)
    return(tf.keras.Model(Inputs,dense2,name='inc'))
