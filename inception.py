import tensorflow as tf

class InceptionBlock(tf.keras.Model):
    def __init__(self,bottleneck_size=32,nb_filters=32):
        super(InceptionBlock, self).__init__()
        self.bottleneck_size = bottleneck_size
        self.nb_filters = nb_filters
        self.bottlneck=tf.keras.layers.Conv1D(filters=bottleneck_size,kernel_size=1,padding='same')
        self.kernel_sizes=[40,20,10]
        self.conv_list=[]
        for kernel_size in self.kernel_sizes:
            self.conv_list.append(tf.keras.layers.Conv1D(filters=self.nb_filters,padding='same',kernel_size=kernel_size))
        self.max_pool=tf.keras.layers.MaxPool1D(pool_size=3,padding='same')
        self.conv_short=tf.keras.layers.Conv1D(filters=self.nb_filters,kernel_size=1,padding='same')
        self.concat=tf.keras.layers.Concatenate()
        self.normal=tf.keras.layers.BatchNormalization()
        self.activation=tf.keras.layers.Activation(activation='relu')
        self.output_dim=(len(self.conv_list)+1)*nb_filters
    def call(self,input):
        print(input.shape)
        x=self.bottlneck(input)
        print(input.shape)
        print(x.shape)
        res_list=[]
        for conv in self.conv_list:
            res_list.append(conv(x))
        short=self.max_pool(input)
        print(short.shape)
        res_list.append(self.conv_short(short))
        print(res_list[-1].shape)
        out=self.concat(res_list)
        out=self.normal(out)
        return self.activation(out)


class Inception(tf.keras.Model):
    def __init__(self,depth_od=6,depth_sc=6):
        super(Inception,self).__init__()
        self.depth_sc=depth_sc
        self.depth_od=depth_od
        self.inception_od=[]
        self.inception_sc=[]
        for d in range(depth_od):
            self.inception_od.append(InceptionBlock())
        for d in range(depth_sc):
            self.inception_sc.append(InceptionBlock())
        self.avg_pool_od=tf.keras.layers.GlobalAveragePooling1D()
        self.avg_pool_sc=tf.keras.layers.GlobalAveragePooling1D()
        self.concat=tf.keras.layers.Concatenate()
        self.dense=tf.keras.layers.Dense(3,activation=tf.nn.softmax)
        self.od_short_convs=[]
        self.od_short_norm=[]
        self.od_short_add=[]
        self.od_short_act=[]
        for i in range(self.depth_od):
            if i%3==2:
                self.od_short_convs.append(tf.keras.layers.Conv1D(filters=self.inception_od[i].output_dim,kernel_size=1,padding='same'))
                self.od_short_act.append(tf.keras.layers.Activation('relu'))
                self.od_short_add.append(tf.keras.layers.Add())
                self.od_short_norm.append(tf.keras.layers.BatchNormalization())
            else:
                self.od_short_norm.append(None)
                self.od_short_act.append(None)
                self.od_short_add.append(None)
                self.od_short_convs.append(None)
        self.sc_short_convs=[]
        self.sc_short_norm=[]
        self.sc_short_add=[]
        self.sc_short_act=[]
        for i in range(self.depth_sc):
            if i%3==2:
                self.sc_short_convs.append(tf.keras.layers.Conv1D(filters=self.inception_sc[i].output_dim,kernel_size=1,padding='same'))
                self.sc_short_act.append(tf.keras.layers.Activation('relu'))
                self.sc_short_add.append(tf.keras.layers.Add())
                self.sc_short_norm.append(tf.keras.layers.BatchNormalization())
            else:
                self.sc_short_norm.append(None)
                self.sc_short_act.append(None)
                self.sc_short_add.append(None)
                self.sc_short_convs.append(None)
    def call(self,inputs):
        od_inp,sc_inp=inputs
        od=od_inp
        sc=sc_inp
        for i,inc in enumerate(self.inception_od):
            od=inc(od)
            if i%3==2:
                conv=self.od_short_convs[i]
                norm=self.od_short_norm[i]
                add=self.od_short_add[i]
                act=self.od_short_act[i]
                new_od_inp=conv(od_inp)
                new_od_inp=norm(new_od_inp)
                od=add([new_od_inp,od])
                od=act(od)
                od_inp=od
        for i,inc in enumerate(self.inception_sc):
            sc=inc(sc)
            if i%3==2:
                conv=self.sc_short_convs[i]
                norm=self.sc_short_norm[i]
                add=self.sc_short_add[i]
                act=self.sc_short_act[i]
                new_sc_inp=conv(sc_inp)
                new_sc_inp=norm(new_sc_inp)
                sc=add([new_sc_inp,sc])
                sc=act(sc)
                sc_inp=sc
        od=self.avg_pool_od(od)
        sc=self.avg_pool_sc(sc)
        x=self.concat([od,sc])
        return self.dense(x)
        

