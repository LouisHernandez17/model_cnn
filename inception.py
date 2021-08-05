import tensorflow as tf

class InceptionBlock(tf.keras.Model):
    def __init__(self,bottleneck_size=32,nb_filters=32):
        super(InceptionBlock, self).__init__()
        self.bottleneck_size = bottleneck_size
        self.nb_filters = nb_filters
        self.bottlneck=tf.keras.layers.Conv1D(filters=bottleneck_size,kernel_size=1,padding='same',use_bias=False)
        self.kernel_sizes=[40,20,10]
        self.conv_list=[]
        for kernel_size in self.kernel_sizes:
            self.conv_list.append(tf.keras.layers.Conv1D(filters=self.nb_filters,padding='same',kernel_size=kernel_size,use_bias=False))
        self.max_pool=tf.keras.layers.MaxPool1D(pool_size=3,padding='same',strides=1)
        self.conv_short=tf.keras.layers.Conv1D(filters=self.nb_filters,kernel_size=1,padding='same',use_bias=False)
        self.concat=tf.keras.layers.Concatenate()
        self.normal=tf.keras.layers.BatchNormalization()
        self.activation=tf.keras.layers.Activation(activation='relu')
        self.output_dim=(len(self.conv_list)+1)*nb_filters
    def call(self,input):
        x=self.bottlneck(input)
        res_list=[]
        for conv in self.conv_list:
            res=conv(x)
            res_list.append(res)
        short=self.max_pool(input)
        res=self.conv_short(short)
        res_list.append(res)
        out=self.concat(res_list)
        out=self.normal(out)
        return self.activation(out)


class Inception(tf.keras.Model):
    def __init__(self,depth=6,n_branches=2):
        super(Inception,self).__init__()
        self.short_name='inc'
        self.n_branches=n_branches
        self.depth=depth
        self.inceptions=[[] for i in range(n_branches)]
        for inception in self.inceptions:
            for d in range(depth):
                inception.append(InceptionBlock())
        self.avg_pools=[tf.keras.layers.GlobalAveragePooling1D() for i in range(n_branches)]
        self.concat=tf.keras.layers.Concatenate()
        self.dense1=tf.keras.layers.Dense(64,activation=tf.nn.relu)
        self.dense2=tf.keras.layers.Dense(128,activation=tf.nn.relu)
        self.dense3=tf.keras.layers.Dense(3,activation=tf.nn.softmax)
        self.short_convs=[[] for i in range(n_branches)]
        self.short_norm=[[] for i in range(n_branches)]
        self.short_add=[[] for i in range(n_branches)]
        self.short_act=[[] for i in range(n_branches)]
        for k in range(n_branches):
            for i in range(self.depth):
                if i%3==2:
                    self.short_convs[k].append(tf.keras.layers.Conv1D(filters=self.inceptions[k][i].output_dim,kernel_size=1,padding='same',use_bias=False))
                    self.short_act[k].append(tf.keras.layers.Activation('relu'))
                    self.short_add[k].append(tf.keras.layers.Add())
                    self.short_norm[k].append(tf.keras.layers.BatchNormalization())
                else:
                    self.short_norm[k].append(None)
                    self.short_act[k].append(None)
                    self.short_add[k].append(None)
                    self.short_convs[k].append(None)
    def call(self,inputs):
        xs=[]
        for k in range(self.n_branches):
            inp=inputs[k].to_tensor()
            for i,inc in enumerate(self.inceptions[k]):
                x=inc(inp)
                if i%3==2:
                    conv=self.short_convs[k][i]
                    norm=self.short_norm[k][i]
                    add=self.short_add[k][i]
                    act=self.short_act[k][i]
                    new_inp=conv(inp)
                    new_inp=norm(new_inp)
                    x=add([new_inp,x])
                    x=act(x)
                    inp=x
            xs.append(self.avg_pools[k](x))
        x=self.concat(xs)
        x=self.dense1(x)
        x=self.dense2(x)
        return(self.dense3(x))
        

