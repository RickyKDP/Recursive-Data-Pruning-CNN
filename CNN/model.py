# -*- coding: utf-8 -*-
#TextCNN: 1. embeddding layers, 2.convolutional layer, 3.max-pooling, 4.softmax layer.
# print("started...")
import tensorflow as tf
import numpy as np

class CNN_Config(object):
  def __init__(self,
                vocab_mat,
                hidden_dropout_prob=0.5,
                batch_size = 100,
                sequence_length = 128,
                num_classes = 4,
                filter_sizes = [0,1,2],
                filter_num_list = [64,64,64],
                learning_rate = 0.01,
                decay_steps = 400,
                decay_rate = 0.6,
                multi_label_flag = False):

        self.vocab_emb = vocab_mat
        self.hidden_dropout_prob = hidden_dropout_prob
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.filter_sizes = filter_sizes
        self.filter_num_list = filter_num_list
        self.learning_rate = learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.multi_label_flag = multi_label_flag

class TextCNN:
    def __init__(self,net_config,round_num,is_training=True,initializer=tf.random_normal_initializer(stddev=0.1),decay_rate_big=0.500):
        """init all hyperparameter here"""
        # set hyperparamter
        self.num_classes = net_config.num_classes
        self.batch_size = net_config.batch_size
        self.sequence_length = net_config.sequence_length
        self.embed_size = net_config.vocab_emb.shape[1]
        self.is_training = is_training
        self.learning_rate = tf.Variable(net_config.learning_rate, trainable=False, name="learning_rate")#ADD learning_rate
        self.learning_rate_decay_half_op = tf.assign(self.learning_rate, self.learning_rate * decay_rate_big)
        self.filter_sizes = net_config.filter_sizes # it is a list of int. e.g. [3,4,5]
        self.num_filters = net_config.filter_num_list
        self.initializer = initializer
        self.num_filters_total = 0
        for item in self.num_filters:
            self.num_filters_total += item  #how many filters totally.
        self.multi_label_flag= net_config.multi_label_flag
#        self.clip_gradients = clip_gradients
        self.round_num =round_num
        # add placeholder (X,label)
        self.input_x = tf.placeholder(tf.int32, [None, None], name="input_x")  # X
        self.x_mask = tf.placeholder(tf.float32,[None, None], name="x_mask")
        self.input_y = tf.placeholder(tf.int32, [None,net_config.num_classes],name="input_y")  # y:[None,num_classes]
#        self.input_y_multilabel = tf.placeholder(tf.float16,[None,self.num_classes], name="input_y_multilabel")  # y:[None,num_classes]. this is for multi-label classification only.
        self.dropout_keep_prob=tf.placeholder(tf.float32,name="dropout_keep_prob")
        self.tst=tf.placeholder(tf.bool)

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step=tf.Variable(0,trainable=False,name="Epoch_Step")
        self.epoch_increment=tf.assign(self.epoch_step,tf.add(self.epoch_step,tf.constant(1)))
#        self.b1 = tf.Variable(tf.ones([self.num_filters]) / 10)
#        self.b2 = tf.Variable(tf.ones([self.num_filters]) / 10)
        self.decay_steps, self.decay_rate = net_config.decay_steps, net_config.decay_rate

        self.instantiate_weights(net_config.vocab_emb)
        self.logits = self.inference() #[None, self.label_size]. main computation graph is here.
        self.possibility=tf.nn.sigmoid(self.logits)
        if not is_training:
            return
        if self.multi_label_flag:print("going to use multi label loss.");self.loss_val = self.loss_multilabel()
        else:print("going to use single label loss.");self.loss_val = self.loss()
        tf.summary.scalar("loss",self.loss_val)
        self.train_op = self.train()
        if not self.multi_label_flag:
            self.predictions = tf.cast(tf.argmax(self.logits, 1, name="predictions"),tf.int32)  # shape:[None,]
            label = tf.cast(tf.argmax(self.input_y, 1,name="label"), tf.int32)
            correct_prediction = tf.equal(self.predictions, label) #tf.argmax(self.logits, 1)-->[batch_size]
            self.accuracy =tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy") # shape=()
            tf.summary.scalar("accuracy",self.accuracy)
        self.merged_summary = tf.summary.merge_all()

    def instantiate_weights(self,embedded_mat):
        """define all weights here"""
        with tf.name_scope("embedding"): # embedding matrix
            self.Embedding = tf.get_variable("Embedding",initializer=embedded_mat,dtype=tf.float32 ) #[vocab_size,embed_size] tf.random_uniform([self.vocab_size, self.embed_size],-1.0,1.0)
            self.W_projection = tf.get_variable("W_projection",shape=[self.num_filters_total, self.num_classes],initializer=self.initializer,dtype=tf.float32) #[embed_size,label_size]
            self.b_projection = tf.get_variable("b_projection",shape=[self.num_classes],dtype=tf.float32)       #[label_size] #ADD 2017.06.09

    def inference(self):
        """main computation graph here: 1.embedding-->2.CONV-BN-RELU-MAX_POOLING-->3.linear classifier"""
        # 1.=====>get emebedding of words in the sentence
        self.embedded_words = tf.nn.embedding_lookup(self.Embedding,self.input_x)#[None,sentence_length,embed_size]
        self.embedded_words = tf.math.multiply(tf.expand_dims(self.x_mask,-1),self.embedded_words)
        self.sentence_embeddings_expanded=tf.expand_dims(self.embedded_words,-1) #[None,sentence_length,embed_size,1). expand dimension so meet input requirement of 2d-conv
 
        # 2.=====>loop each filter size. for each filter, do:convolution-pooling layer(a.create filters,b.conv,c.apply nolinearity,d.max-pooling)--->
        # you can use:tf.nn.conv2d;tf.nn.relu;tf.nn.max_pool; feature shape is 4-d. feature is a new variable
        pooled_outputs = []
        max_position = []
        for i,filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("convolution-pooling-%s" %filter_size):
                # ====>a.create filter
                filter_CNN=tf.get_variable("filter-%s"%filter_size,[filter_size,self.embed_size,1,self.num_filters[i]],initializer=self.initializer,dtype = tf.float32)
#                if self.round_num != 0:
#                    filter_new=tf.get_variable("filter_new-%s"%filter_size,[filter_size,self.embed_size,1,self.new_filter_num],initializer=self.initializer,dtype = tf.float32)
#                    conv_new=tf.nn.conv2d(self.sentence_embeddings_expanded, filter_new, strides=[1,1,1,1], padding="VALID",name="conv")
#                    b_new=tf.get_variable("b_new-%s"%filter_size,[self.new_filter_num])
#                    h_new=tf.nn.relu(tf.nn.bias_add(conv_new,b_new),"relu")
#                    pooled_new,pos_n=tf.nn.max_pool_with_argmax(h_new, ksize=[1,self.sequence_length-filter_size+1,1,1], strides=[1,1,1,1], padding='VALID',name="pool")
#                    pos_n = tf.reshape(pos_n,[self.batch_size,self.new_filter_num])
#                    minus_matrix_new = []
#                    for ii in range(self.batch_size):
#                        minus_matrix_new.append(tf.expand_dims(tf.range(self.new_filter_num,dtype = tf.int64),0))
#                    minus_matrix_new_ = tf.concat(minus_matrix_new,0)
#                    pos_n = (pos_n-minus_matrix_new_)/self.new_filter_num


                # ====>b.conv operation: conv2d===>computes a 2-D convolution given 4-D `input` and `filter` tensors.
                #Conv.Input: given an input tensor of shape `[batch, in_height, in_width, in_channels]` and a filter / kernel tensor of shape `[filter_height, filter_width, in_channels, out_channels]`
                #Conv.Returns: A `Tensor`. Has the same type as `input`.
                #         A 4-D tensor. The dimension order is determined by the value of `data_format`, see below for details.
                #1)each filter with conv2d's output a shape:[1,sequence_length-filter_size+1,1,1];2)*num_filters--->[1,sequence_length-filter_size+1,1,num_filters];3)*batch_size--->[batch_size,sequence_length-filter_size+1,1,num_filters]
                #input data format:NHWC:[batch, height, width, channels];output:4-D
                conv=tf.nn.conv2d(self.sentence_embeddings_expanded, filter_CNN, strides=[1,1,1,1], padding="VALID",name="conv") #shape:[batch_size,sequence_length - filter_size + 1,1,num_filters]

#                conv,self.update_ema=self.batchnorm(conv,self.tst, self.iter, self.b1) # TODO remove it temp
                # ====>c. apply nolinearity
                b=tf.get_variable("b-%s"%filter_size,[self.num_filters[i]]) #ADD 2017-06-09
                h=tf.nn.relu(tf.nn.bias_add(conv,b),"relu") #shape:[batch_size,sequence_length - filter_size + 1,1,num_filters]. tf.nn.bias_add:adds `bias` to `value`
                # ====>. max-pooling.  value: A 4-D `Tensor` with shape `[batch, height, width, channels]
                #                  ksize: A list of ints that has length >= 4.  The size of the window for each dimension of the input tensor.
                #                  strides: A list of ints that has length >= 4.  The stride of the sliding window for each dimension of the input tensor.
                pooled,max_position_=tf.nn.max_pool_with_argmax(h, ksize=[1,self.sequence_length-filter_size+1,1,1], strides=[1,1,1,1], padding='VALID',name="pool")#shape:[batch_size, 1, 1, num_filters].max_pool:performs the max pooling on the input.
                max_position_ = tf.reshape(max_position_,[self.batch_size,self.num_filters[i]])

                minus_matrix_ = []
                for ii in range(self.batch_size):
                    minus_matrix_.append(tf.expand_dims(tf.range(self.num_filters[i],dtype = tf.int64),0))
                minus_matrix = tf.concat(minus_matrix_,0)
                max_position_ = (max_position_-minus_matrix)/self.num_filters[i]

                max_position.append(max_position_)
                pooled_outputs.append(pooled)

        # orth filter set initialization#
#        filter_CNN_orth=tf.get_variable("orth_filter",[1,self.embed_size,1,64],initializer=self.initializer,dtype = tf.float32)
#        conv_orth=tf.nn.conv2d(self.sentence_embeddings_expanded, filter_CNN_orth, strides=[1,1,1,1], padding="VALID",name="conv_orth")
#        b_orth=tf.get_variable("b_orth",[64])
#        h_orth=tf.nn.relu(tf.nn.bias_add(conv_orth,b_orth),"relu")
#        pooled_orth,max_position_=tf.nn.max_pool_with_argmax(h, ksize=[1,self.sequence_length-filter_size+1,1,1], strides=[1,1,1,1], padding='VALID',name="pool")
#        pooled_outputs.append(pooled_orth)
        #################################

        # 3.=====>combine all pooled features, and flatten the feature.output' shape is a [1,None]
        #e.g. >>> x1=tf.ones([3,3]);x2=tf.ones([3,3]);x=[x1,x2]
        #         x12_0=tf.concat(x,0)---->x12_0' shape:[6,3]
        #         x12_1=tf.concat(x,1)---->x12_1' shape;[3,6]
        self.p_pool=tf.concat(max_position,1)
        self.p_pool_flat=tf.reshape(self.p_pool,[-1,self.num_filters_total])
        self.h_pool=tf.concat(pooled_outputs,3) #shape:[batch_size, 1, 1, num_filters_total]. tf.concat=>concatenates tensors along one dimension.where num_filters_total=num_filters_1+num_filters_2+num_filters_3
        self.h_pool_flat=tf.reshape(self.h_pool,[-1,self.num_filters_total]) #shape should be:[None,num_filters_total]. here this operation has some result as tf.sequeeze().e.g. x's shape:[3,3];tf.reshape(-1,x) & (3, 3)---->(1,9)

        #4.=====>add dropout: use tf.nn.dropout
        with tf.name_scope("dropout"):
            self.h_drop=tf.nn.dropout(self.h_pool_flat,keep_prob=self.dropout_keep_prob) #[None,num_filters_total]
        self.h_drop=tf.layers.dense(self.h_drop,self.num_filters_total,activation=tf.nn.tanh,use_bias=True)
        #5. logits(use linear layer)and predictions(argmax)
        with tf.name_scope("output"):
            logits = tf.matmul(self.h_drop,self.W_projection) + self.b_projection  #shape:[None, self.num_classes]==tf.matmul([None,self.embed_size],[self.embed_size,self.num_classes])
        return logits

    def batchnorm(self,Ylogits, is_test, iteration, offset, convolutional=False): #check:https://github.com/martin-gorner/tensorflow-mnist-tutorial/blob/master/mnist_4.1_batchnorm_five_layers_relu.py#L89
        """
        batch normalization: keep moving average of mean and variance. use it as value for BN when training. when prediction, use value from that batch.
        :param Ylogits:
        :param is_test:
        :param iteration:
        :param offset:
        :param convolutional:
        :return:
        """
        exp_moving_avg = tf.train.ExponentialMovingAverage(0.999,iteration)  # adding the iteration prevents from averaging across non-existing iterations
        bnepsilon = 1e-5
        if convolutional:
            mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
        else:
            mean, variance = tf.nn.moments(Ylogits, [0])
        update_moving_averages = exp_moving_avg.apply([mean, variance])
        m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
        v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
        Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)
        return Ybn, update_moving_averages

    def loss_multilabel(self,l2_lambda=0.0001): #0.0001#this loss function is for multi-label classification
        with tf.name_scope("loss"):
            #input: `logits` and `labels` must have the same shape `[batch_size, num_classes]`
            #output: A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the softmax cross entropy loss.
            #input_y:shape=(?, 1999); logits:shape=(?, 1999)
            # let `x = logits`, `z = labels`.  The logistic loss is:z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y_multilabel, logits=self.logits);#losses=tf.nn.softmax_cross_entropy_with_logits(labels=self.input__y,logits=self.logits)
            #losses=-self.input_y_multilabel*tf.log(self.logits)-(1-self.input_y_multilabel)*tf.log(1-self.logits)
            print("sigmoid_cross_entropy_with_logits.losses:",losses) #shape=(?, 1999).
            losses=tf.reduce_sum(losses,axis=1) #shape=(?,). loss for all data in the batch
            loss=tf.reduce_mean(losses)         #shape=().   average loss in the batch

            self.l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            loss=loss+self.l2_losses
        return loss

    def loss(self,l2_lambda=0.0001):#0.001
        with tf.name_scope("loss"):
            #input: `logits`:[batch_size, num_classes], and `labels`:[batch_size]
            #output: A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the softmax cross entropy loss.
#            self.losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits);
            #sigmoid_cross_entropy_with_logits.#
            losses=tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.input_y,logits=self.logits)
            #print("1.sparse_softmax_cross_entropy_with_logits.losses:",losses) # shape=(?,)
            loss=tf.reduce_mean(losses)#print("4.loss.loss:", loss) #shape=()
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'Embedding' not in v.name]) * l2_lambda
            loss=loss+l2_losses
        return loss

    def train(self):
        """based on the loss, use SGD to update parameter"""
#        learning_rate_Normal = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,self.decay_rate, staircase=True)
        if self.round_num != 0:
            learning_rate_Conv = self.learning_rate*0.1
            variable_list = tf.trainable_variables()
            var_fine_tuning = []
            var_train = []

            for var in variable_list:
                if "filter-" in var.name:
                    var_fine_tuning.append(var)
                elif "b-" in var.name:
                    var_fine_tuning.append(var)
                else:
                    var_train.append(var)

            print(var_fine_tuning)
            print(var_train)
            train_op1 = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.loss_val,var_list=var_train)
            train_op2 = tf.train.AdagradOptimizer(learning_rate_Conv).minimize(self.loss_val,var_list=var_fine_tuning)
            train_op = tf.group(train_op1,train_op2)
        else:
            train_op = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.loss_val)
#        train_op = tf.contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step,learning_rate=learning_rate, optimizer="Adam",clip_gradients=self.clip_gradients)
        return train_op

#test started. toy task: given a sequence of data. compute it's label: sum of its previous element,itself and next element greater than a threshold, it's label is 1,otherwise 0.
#e.g. given inputs:[1,0,1,1,0]; outputs:[0,1,1,1,0].
#invoke test() below to test the model in this toy task.
def test():
    #below is a function test; if you use this for text classifiction, you need to transform sentence to indices of vocabulary first. then feed data to the graph.
    num_classes=5
    learning_rate=0.001
    batch_size=8
    decay_steps=1000
    decay_rate=0.95
    sequence_length=5
    vocab_size=10000
    embed_size=100
    is_training=True
    dropout_keep_prob=1.0 #0.5
    filter_sizes=[2,3,4]
    num_filters=128
    multi_label_flag=True
    textRNN=TextCNN(filter_sizes,num_filters,num_classes, learning_rate, batch_size, decay_steps, decay_rate,sequence_length,vocab_size,embed_size,is_training,multi_label_flag=multi_label_flag)
    with tf.Session() as sess:
       sess.run(tf.global_variables_initializer())
       for i in range(500):
           input_x=np.random.randn(batch_size,sequence_length) #[None, self.sequence_length]
           input_x[input_x>=0]=1
           input_x[input_x <0] = 0
           input_y_multilabel=get_label_y(input_x)
           loss,possibility,W_projection_value,_=sess.run([textRNN.loss_val,textRNN.possibility,textRNN.W_projection,textRNN.train_op],
                                                    feed_dict={textRNN.input_x:input_x,textRNN.input_y_multilabel:input_y_multilabel,
                                                               textRNN.dropout_keep_prob:dropout_keep_prob,textRNN.tst:False})
           print(i,"loss:",loss,"-------------------------------------------------------")
           print("label:",input_y_multilabel);#print("possibility:",possibility)

def get_label_y(input_x):
    length=input_x.shape[0]
    input_y=np.zeros((input_x.shape))
    for i in range(length):
        element=input_x[i,:] #[5,]
        result=compute_single_label(element)
        input_y[i,:]=result
    return input_y

def compute_single_label(listt):
    result=[]
    length=len(listt)
    for i,e in enumerate(listt):
        previous=listt[i-1] if i>0 else 0
        current=listt[i]
        next=listt[i+1] if i<length-1 else 0
        summ=previous+current+next
        if summ>=2:
            summ=1
        else:
            summ=0
        result.append(summ)
    return result


#test()
