import tensorflow as tf
import numpy as np
from CNN.model import *
import os
import random
import pandas as pd
import numpy as np
from CNN.util import *
import time

FLAGS=tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("traning_data_path","../data/sample_multiple_label.txt","path of traning data.") #../data/sample_multiple_label.txt
tf.app.flags.DEFINE_integer("vocab_size",100001,"maximum vocab size.")

tf.app.flags.DEFINE_float("learning_rate",0.05,"learning rate")
tf.app.flags.DEFINE_integer("batch_size", 200, "Batch size for training/evaluating.") #批处理的大小 32-->128
tf.app.flags.DEFINE_integer("decay_steps", 40000, "how many steps before decay learning rate.") #6000批处理的大小 32-->128
tf.app.flags.DEFINE_float("decay_rate", 0.7, "Rate of decay for learning rate.") #0.65一次衰减多少
tf.app.flags.DEFINE_string("ckpt_dir","./CNN/CNN_checkpoint/base/","checkpoint location for the model")
tf.app.flags.DEFINE_integer("sentence_len",1000,"max sentence length")
tf.app.flags.DEFINE_integer("embed_size",300,"embedding size")
tf.app.flags.DEFINE_boolean("is_training",True,"is traning.true:tranining,false:testing/inference")
tf.app.flags.DEFINE_integer("num_epochs",100,"number of epochs to run.")
tf.app.flags.DEFINE_integer("validate_every", 1, "Validate every validate_every epochs.") #每10轮做一次验证
tf.app.flags.DEFINE_boolean("use_embedding",True,"whether to use embedding or not.")
tf.app.flags.DEFINE_integer("num_filter", 128 , "number of filters") #256--->512
tf.app.flags.DEFINE_string("word2vec_model_path","word2vec-title-desc.bin","word2vec's vocabulary and vectors")
tf.app.flags.DEFINE_string("name_scope","cnn","name scope value.")
tf.app.flags.DEFINE_boolean("multi_label_flag",False,"use multi label or single label.")
tf.app.flags.DEFINE_integer("num_class",4,"number of classes in this dataset.")
filter_sizes=[1,2,3]

#1.load data(X:list of lint,y:int). 2.create session. 3.feed data. 4.training (5.validation) ,(6.prediction)
def CNN(data,vocab_emb,num_filters,ref_str,preset_para,lr,round_num,t_ix):
    trainX, trainY, testX, testY = data[0],data[3],data[2],data[5]
    evalX, evalY = data[1],data[4]
 

    max_len = 0
    for item in trainX:
        if len(item)>max_len:
            max_len = len(item)

    FLAGS.sentence_len = max_len
    print(FLAGS.sentence_len)

    #print some message for debug purpose
    print("length of training data:",len(trainX),";length of testing data:",len(testX))

    vocab_size = vocab_emb.shape[0]

    test_max = 0.0

    CNN_config = CNN_Config(
        vocab_mat = vocab_emb,
        hidden_dropout_prob = 0.5,
        batch_size = FLAGS.batch_size,
        sequence_length = FLAGS.sentence_len,
        num_classes = FLAGS.num_class,
        filter_sizes = filter_sizes,
        filter_num_list = num_filters,
        learning_rate = lr,
        decay_steps = FLAGS.decay_steps,
        decay_rate = FLAGS.decay_rate,
        multi_label_flag = FLAGS.multi_label_flag)

    #2.create session.
    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        #Instantiate Model
        textCNN=TextCNN(net_config = CNN_config,
                        round_num = round_num,
                        is_training = FLAGS.is_training)
#        all_v = tf.global_variables()
#        for i in all_v:
#            print(i)

        writer_train = tf.summary.FileWriter(f"./CNN/log_train/train_{round_num}")


        #Initialize Save
        saver=tf.train.Saver()
#        if os.path.exists(FLAGS.ckpt_dir+"checkpoint"):
#            print("Restoring Variables from Checkpoint.")
#            saver.restore(sess,tf.train.latest_checkpoint(FLAGS.ckpt_dir))
#            #for i in range(3): #decay learning rate if necessary.
#            #    print(i,"Going to decay learning rate by half.")
#            #    sess.run(textCNN.learning_rate_decay_half_op)
#        else:

        print('Initializing Variables')
        sess.run(tf.global_variables_initializer())

        #####################preset para###########################
        if round_num != 0:
            for filter_ix in range(3):
                filter_ix_ = filter_ix+1
                filter_para = tf.assign(tf.get_default_graph().get_tensor_by_name(f"filter-{filter_ix_}:0"),preset_para[0][filter_ix])
                b_para = tf.assign(tf.get_default_graph().get_tensor_by_name(f"b-{filter_ix_}:0"),preset_para[1][filter_ix])
                sess.run(filter_para)
                sess.run(b_para)
            print("********************************init_over***********************************")


############################ Training ############################
        curr_epoch=sess.run(textCNN.epoch_step)
        #3.feed data & training
        number_of_training_data=len(trainX)
        batch_size=FLAGS.batch_size
        epoch_test = []
        train_para ={"loss":[],"acc":[],"time":[]}
        test_para ={"loss":[],"acc":[],"time":[]}

        for epoch in range(curr_epoch,FLAGS.num_epochs):
            loss,accuracy =  0.0, 0.0
            batch_index = get_minibatches_idx(len(trainX),FLAGS.batch_size,shuffle=True)
            for counter,train_index in batch_index:
                input_x = [trainX[i] for i in train_index]
                input_y = [trainY[i] for i in train_index]
                input_x,x_mask = prepare_data_for_emb(input_x,FLAGS.sentence_len)

                feed_dict = {textCNN.input_x: input_x,textCNN.x_mask: x_mask, textCNN.dropout_keep_prob: 0.5,textCNN.tst: not FLAGS.is_training}

                feed_dict[textCNN.input_y] = input_y

#                print(sess.run(textCNN.loss_val,feed_dict))

                curr_loss,curr_acc,_,s_train =sess.run([textCNN.loss_val,textCNN.accuracy,textCNN.train_op,textCNN.merged_summary],feed_dict)
                writer_train.add_summary(s_train,counter)
                loss,accuracy=loss+curr_loss,accuracy+curr_acc
                train_para["loss"].append(curr_loss)
                train_para["acc"].append(curr_acc)
                if counter%5==0:
                    print("Epoch %d\tBatch %d\tTrain Loss:%.3f\tTrain Accuracy:%.5f" %(epoch,counter,loss/float(counter),accuracy/float(counter)))


######################################## Testing ##################################################

            test_loss_val,test_accuracy = 0.0,0.0
            batch_index = get_minibatches_idx(len(testX),1,shuffle=False)
            t0 = time.time()
            for t_counter,test_index in batch_index:
                input_x = [testX[i] for i in test_index]
                input_y = [testY[i] for i in test_index]
                input_x,x_mask = prepare_data_for_emb(input_x,FLAGS.sentence_len)
                feed_dict = {textCNN.input_x: input_x,textCNN.x_mask: x_mask, textCNN.dropout_keep_prob: 1,textCNN.tst: not FLAGS.is_training}
                feed_dict[textCNN.input_y] = input_y
                test_loss,test_acc = sess.run([textCNN.loss_val,textCNN.accuracy],feed_dict)
                test_loss_val,test_accuracy = test_loss_val+test_loss, test_accuracy+test_acc

            t1 = time.time()
            print("Epoch %d Validation Loss:%.5f\tValidation Accuracy:%.5f" % (epoch, test_loss_val/float(t_counter), test_accuracy/float(t_counter)))
            test_para["loss"].append(test_loss_val/float(t_counter))
            test_para["acc"].append(test_accuracy/float(t_counter))
            test_para["time"].append(t1-t0)
            if test_accuracy/float(t_counter) > test_max:
                test_max = test_accuracy/float(t_counter)


            # save model to checkpoint
            save_path = FLAGS.ckpt_dir +f"{t_ix}_{round_num}_" + f"model.ckpt"
            saver.save(sess, save_path, global_step=epoch)
            pd.to_pickle([train_para,test_para],f"./analyze/ag_news/CNN_{round_num}_{t_ix}.pkl")

#        #########################################################################################################
#            #epoch increment
#            print("going to increment epoch counter....")
#            sess.run(textCNN.epoch_increment)

        ############################embedding calculation#######################################
        Convolution_output = []
        Position_output = []
        batch_index = get_minibatches_idx(len(trainX),FLAGS.batch_size,shuffle=False)
        for counter,train_index in batch_index:
            input_x = [trainX[i] for i in train_index]
            input_y = [trainY[i] for i in train_index]
            input_x,x_mask = prepare_data_for_emb(input_x,FLAGS.sentence_len)

            feed_dict = {textCNN.input_x: input_x,textCNN.x_mask: x_mask, textCNN.dropout_keep_prob:0,textCNN.tst: not FLAGS.is_training}
            feed_dict[textCNN.input_y] = input_y

            _,Convolution_output_,Position_output_ = sess.run([textCNN.loss_val,textCNN.h_pool_flat,textCNN.p_pool_flat],feed_dict)
            Convolution_output.append(Convolution_output_)
            Position_output.append(Position_output_)
        Convolution_output = np.concatenate(Convolution_output,axis=0)
        Position_output = np.concatenate(Position_output,axis=0)

        Convolution_output_eval = []
        Position_output_eval = []
        batch_index = get_minibatches_idx(len(evalX),FLAGS.batch_size,shuffle=False)
        for counter,eval_index in batch_index:
            input_x = [evalX[i] for i in eval_index]
            input_y = [evalY[i] for i in eval_index]
            input_x,x_mask = prepare_data_for_emb(input_x,FLAGS.sentence_len)
            feed_dict = {textCNN.input_x: input_x,textCNN.x_mask: x_mask, textCNN.dropout_keep_prob:0,textCNN.tst: not FLAGS.is_training}
            feed_dict[textCNN.input_y] = input_y

            _,Convolution_output_,Position_output_ = sess.run([textCNN.loss_val,textCNN.h_pool_flat,textCNN.p_pool_flat],feed_dict)
            Convolution_output_eval.append(Convolution_output_)
            Position_output_eval.append(Position_output_)
        Convolution_output_eval = np.concatenate(Convolution_output_eval,axis=0)
        Position_output_eval = np.concatenate(Position_output_eval,axis=0)

        ##############################embedding end#############################################


        ########### filter para  #############
        filter_1 = tf.get_default_graph().get_tensor_by_name("filter-1:0")
        filter_1_b = tf.get_default_graph().get_tensor_by_name("b-1:0")
        filter_2 = tf.get_default_graph().get_tensor_by_name("filter-2:0")
        filter_2_b = tf.get_default_graph().get_tensor_by_name("b-2:0")
        filter_3 = tf.get_default_graph().get_tensor_by_name("filter-3:0")
        filter_3_b = tf.get_default_graph().get_tensor_by_name("b-3:0")
        embedding = tf.get_default_graph().get_tensor_by_name("Embedding_1:0")
#        if round_num != 0:
#            filter_new_1 = tf.get_default_graph().get_tensor_by_name("filter_new-1:0")
#            filter_new_1_b = tf.get_default_graph().get_tensor_by_name("b_new-1:0")
#            filter_new_2 = tf.get_default_graph().get_tensor_by_name("filter_new-2:0")
#            filter_new_2_b = tf.get_default_graph().get_tensor_by_name("b_new-2:0")
#            filter_new_3 = tf.get_default_graph().get_tensor_by_name("filter_new-3:0")
#            filter_new_3_b = tf.get_default_graph().get_tensor_by_name("b_new-3:0")
#            filter_parameter = dict()
#            filter_parameter["filter_1"] = [np.concatenate([sess.run(filter_1),sess.run(filter_new_1)],axis=3),np.concatenate([sess.run(filter_1_b),sess.run(filter_new_1_b)],axis=0)]
#            filter_parameter["filter_2"] = [np.concatenate([sess.run(filter_2),sess.run(filter_new_2)],axis=3),np.concatenate([sess.run(filter_2_b),sess.run(filter_new_2_b)],axis=0)]
#            filter_parameter["filter_3"] = [np.concatenate([sess.run(filter_3),sess.run(filter_new_3)],axis=3),np.concatenate([sess.run(filter_3_b),sess.run(filter_new_3_b)],axis=0)]
        filter_parameter = dict()
        filter_parameter["filter_1"] = sess.run([filter_1,filter_1_b])
        filter_parameter["filter_2"] = sess.run([filter_2,filter_2_b])
        filter_parameter["filter_3"] = sess.run([filter_3,filter_3_b])
        filter_parameter["embedding"] = sess.run(embedding)
        dense_layer_W_tf = tf.get_default_graph().get_tensor_by_name("dense/kernel:0")
        softmax_layer_W_tf = tf.get_default_graph().get_tensor_by_name("W_projection:0")
        dense_layer_W,softmax_layer_W = sess.run([dense_layer_W_tf,softmax_layer_W_tf])


    tf.reset_default_graph()

    return data,[Convolution_output,Convolution_output_eval],[Position_output,Position_output_eval],[dense_layer_W,softmax_layer_W],filter_parameter,CNN_config,test_para

def CNN_load(data,vocab_emb,num_filters,ref_str,preset_para,lr,round_num,model_dir):
    trainX, trainY, testX, testY = data[0],data[3],data[2],data[5]
    evalX, evalY = data[1],data[4]

    max_len = 0
    for item in trainX:
        if len(item)>max_len:
            max_len = len(item)

    FLAGS.sentence_len = max_len

    CNN_config = CNN_Config(
        vocab_mat = vocab_emb,
        hidden_dropout_prob = 0.5,
        batch_size = FLAGS.batch_size,
        sequence_length = FLAGS.sentence_len,
        num_classes = FLAGS.num_class,
        filter_sizes = filter_sizes,
        filter_num_list = num_filters,
        learning_rate = lr,
        decay_steps = FLAGS.decay_steps,
        decay_rate = FLAGS.decay_rate,
        multi_label_flag = FLAGS.multi_label_flag)

    if preset_para[2] == None:
        new_filter_num = 0
    else:
        new_filter_num = preset_para[2][0].shape[-1]
    #2.create session.
    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        #Instantiate Model
        textCNN=TextCNN(net_config = CNN_config,
                        round_num = round_num,
                        new_filter_num = new_filter_num,
                        is_training = FLAGS.is_training)

        print("Restoring Variables from Checkpoint.")
        saver=tf.train.Saver()
        saver.restore(sess,tf.train.latest_checkpoint(model_dir))
        ############################embedding calculation#######################################
        Convolution_output = []
        Position_output = []
        batch_index = get_minibatches_idx(len(trainX),FLAGS.batch_size,shuffle=False)
        for counter,train_index in batch_index:
            input_x = [trainX[i] for i in train_index]
            input_y = [trainY[i] for i in train_index]
            input_x,x_mask = prepare_data_for_emb(input_x,FLAGS.sentence_len)

            feed_dict = {textCNN.input_x: input_x,textCNN.x_mask: x_mask, textCNN.dropout_keep_prob:0,textCNN.tst: not FLAGS.is_training}
            feed_dict[textCNN.input_y] = input_y

            _,Convolution_output_,Position_output_ = sess.run([textCNN.loss_val,textCNN.h_pool_flat,textCNN.p_pool_flat],feed_dict)
            Convolution_output.append(Convolution_output_)
            Position_output.append(Position_output_)
        Convolution_output = np.concatenate(Convolution_output,axis=0)
        Position_output = np.concatenate(Position_output,axis=0)

        Convolution_output_eval = []
        Position_output_eval = []
        batch_index = get_minibatches_idx(len(evalX),FLAGS.batch_size,shuffle=False)
        for counter,eval_index in batch_index:
            input_x = [evalX[i] for i in eval_index]
            input_y = [evalY[i] for i in eval_index]
            input_x,x_mask = prepare_data_for_emb(input_x,FLAGS.sentence_len)
            feed_dict = {textCNN.input_x: input_x,textCNN.x_mask: x_mask, textCNN.dropout_keep_prob:0,textCNN.tst: not FLAGS.is_training}
            feed_dict[textCNN.input_y] = input_y

            _,Convolution_output_,Position_output_ = sess.run([textCNN.loss_val,textCNN.h_pool_flat,textCNN.p_pool_flat],feed_dict)
            Convolution_output_eval.append(Convolution_output_)
            Position_output_eval.append(Position_output_)
        Convolution_output_eval = np.concatenate(Convolution_output_eval,axis=0)
        Position_output_eval = np.concatenate(Position_output_eval,axis=0)

        ########### filter para  #############
        filter_1 = tf.get_default_graph().get_tensor_by_name("filter-1:0")
        filter_1_b = tf.get_default_graph().get_tensor_by_name("b-1:0")
        filter_2 = tf.get_default_graph().get_tensor_by_name("filter-2:0")
        filter_2_b = tf.get_default_graph().get_tensor_by_name("b-2:0")
        filter_3 = tf.get_default_graph().get_tensor_by_name("filter-3:0")
        filter_3_b = tf.get_default_graph().get_tensor_by_name("b-3:0")
        embedding = tf.get_default_graph().get_tensor_by_name("Embedding_1:0")
        if round_num != 0:
            filter_new_1 = tf.get_default_graph().get_tensor_by_name("filter_new-1:0")
            filter_new_1_b = tf.get_default_graph().get_tensor_by_name("b_new-1:0")
            filter_new_2 = tf.get_default_graph().get_tensor_by_name("filter_new-2:0")
            filter_new_2_b = tf.get_default_graph().get_tensor_by_name("b_new-2:0")
            filter_new_3 = tf.get_default_graph().get_tensor_by_name("filter_new-3:0")
            filter_new_3_b = tf.get_default_graph().get_tensor_by_name("b_new-3:0")
            filter_parameter = dict()
            filter_parameter["filter_1"] = [np.concatenate([sess.run(filter_1),sess.run(filter_new_1)],axis=3),np.concatenate([sess.run(filter_1_b),sess.run(filter_new_1_b)],axis=0)]
            filter_parameter["filter_2"] = [np.concatenate([sess.run(filter_2),sess.run(filter_new_2)],axis=3),np.concatenate([sess.run(filter_2_b),sess.run(filter_new_2_b)],axis=0)]
            filter_parameter["filter_3"] = [np.concatenate([sess.run(filter_3),sess.run(filter_new_3)],axis=3),np.concatenate([sess.run(filter_3_b),sess.run(filter_new_3_b)],axis=0)]
        else:
            filter_parameter = dict()
            filter_parameter["filter_1"] = sess.run([filter_1,filter_1_b])
            filter_parameter["filter_2"] = sess.run([filter_2,filter_2_b])
            filter_parameter["filter_3"] = sess.run([filter_3,filter_3_b])
        filter_parameter["embedding"] = sess.run(embedding)

    tf.reset_default_graph()

    return data,[Convolution_output,Convolution_output_eval],[Position_output,Position_output_eval],filter_parameter,CNN_config






# 在验证集上做验证，报告损失、精确度
def do_eval(sess,textCNN,evalX,evalY,iteration,num_classes):
    number_examples=len(evalX)
    eval_loss,eval_counter,eval_f1_score,eval_p,eval_r,eval_acc=0.0,0,0.0,0.0,0.0,0.0
    batch_size=1
    label_dict_confuse_matrix=init_label_dict(num_classes)
    for start,end in zip(range(0,number_examples,batch_size),range(batch_size,number_examples,batch_size)):
        feed_dict = {textCNN.input_x: evalX[start:end], textCNN.dropout_keep_prob: 1.0,textCNN.iter: iteration,textCNN.tst: True}
        if not FLAGS.multi_label_flag:
            feed_dict[textCNN.input_y] = evalY[start:end]
        else:
            feed_dict[textCNN.input_y_multilabel]=evalY[start:end]

        curr_eval_loss, curr_eval_acc, logits= sess.run([textCNN.loss_val,textCNN.accuracy,textCNN.logits],feed_dict)#curr_eval_acc--->textCNN.accuracy
#        predict_y = get_label_using_logits(logits[0])
#        target_y= get_target_label_short(evalY[start:end][0])
#        f1_score,p,r=compute_f1_score(list(label_list_top5), evalY[start:end][0])
#        label_dict_confuse_matrix=compute_confuse_matrix(target_y, predict_y, label_dict_confuse_matrix)
        eval_loss,eval_acc,eval_counter=eval_loss+curr_eval_loss,eval_acc+curr_eval_acc,eval_counter+1

#    f1_micro_accusation,f1_macro_accusation=compute_micro_macro(label_dict_confuse_matrix) #label_dict_accusation is a dict, key is: accusation,value is: (TP,FP,FN). where TP is number of True Positive
#    f1_score=(f1_micro_accusation+f1_macro_accusation)/2.0
    return eval_loss/float(eval_counter),eval_acc/float(eval_counter)

#######################################
def compute_f1_score(predict_y,eval_y):
    """
    compoute f1_score.
    :param logits: [batch_size,label_size]
    :param evalY: [batch_size,label_size]
    :return:
    """
    f1_score=0.0
    p_5=0.0
    r_5=0.0
    return f1_score,p_5,r_5

def compute_f1_score_removed(label_list_top5,eval_y):
    """
    compoute f1_score.
    :param logits: [batch_size,label_size]
    :param evalY: [batch_size,label_size]
    :return:
    """
    num_correct_label=0
    eval_y_short=get_target_label_short(eval_y)
    for label_predict in label_list_top5:
        if label_predict in eval_y_short:
            num_correct_label=num_correct_label+1
    #P@5=Precision@5
    num_labels_predicted=len(label_list_top5)
    all_real_labels=len(eval_y_short)
    p_5=num_correct_label/num_labels_predicted
    #R@5=Recall@5
    r_5=num_correct_label/all_real_labels
    f1_score=2.0*p_5*r_5/(p_5+r_5+0.000001)
    return f1_score,p_5,r_5

random_number=300
def compute_confuse_matrix(target_y,predict_y,label_dict,name='default'):
    """
    compute true postive(TP), false postive(FP), false negative(FN) given target lable and predict label
    :param target_y:
    :param predict_y:
    :param label_dict {label:(TP,FP,FN)}
    :return: macro_f1(a scalar),micro_f1(a scalar)
    """
    #1.get target label and predict label
    if random.choice([x for x in range(random_number)]) ==1:
        print(name+".target_y:",target_y,";predict_y:",predict_y) #debug purpose

    #2.count number of TP,FP,FN for each class
    y_labels_unique=[]
    y_labels_unique.extend(target_y)
    y_labels_unique.extend(predict_y)
    y_labels_unique=list(set(y_labels_unique))
    for i,label in enumerate(y_labels_unique): #e.g. label=2
        TP, FP, FN = label_dict[label]
        if label in predict_y and label in target_y:#predict=1,truth=1 (TP)
            TP=TP+1
        elif label in predict_y and label not in target_y:#predict=1,truth=0(FP)
            FP=FP+1
        elif label not in predict_y and label in target_y:#predict=0,truth=1(FN)
            FN=FN+1
        label_dict[label] = (TP, FP, FN)
    return label_dict

def compute_micro_macro(label_dict):
    """
    compute f1 of micro and macro
    :param label_dict:
    :return: f1_micro,f1_macro: scalar, scalar
    """
    f1_micro = compute_f1_micro_use_TFFPFN(label_dict)
    f1_macro= compute_f1_macro_use_TFFPFN(label_dict)
    return f1_micro,f1_macro

def compute_TF_FP_FN_micro(label_dict):
    """
    compute micro FP,FP,FN
    :param label_dict_accusation: a dict. {label:(TP, FP, FN)}
    :return:TP_micro,FP_micro,FN_micro
    """
    TP_micro,FP_micro,FN_micro=0.0,0.0,0.0
    for label,tuplee in label_dict.items():
        TP,FP,FN=tuplee
        TP_micro=TP_micro+TP
        FP_micro=FP_micro+FP
        FN_micro=FN_micro+FN
    return TP_micro,FP_micro,FN_micro
def compute_f1_micro_use_TFFPFN(label_dict):
    """
    compute f1_micro
    :param label_dict: {label:(TP,FP,FN)}
    :return: f1_micro: a scalar
    """
    TF_micro_accusation, FP_micro_accusation, FN_micro_accusation =compute_TF_FP_FN_micro(label_dict)
    f1_micro_accusation = compute_f1(TF_micro_accusation, FP_micro_accusation, FN_micro_accusation,'micro')
    return f1_micro_accusation

def compute_f1_macro_use_TFFPFN(label_dict):
    """
    compute f1_macro
    :param label_dict: {label:(TP,FP,FN)}
    :return: f1_macro
    """
    f1_dict= {}
    num_classes=len(label_dict)
    for label, tuplee in label_dict.items():
        TP,FP,FN=tuplee
        f1_score_onelabel=compute_f1(TP,FP,FN,'macro')
        f1_dict[label]=f1_score_onelabel
    f1_score_sum=0.0
    for label,f1_score in f1_dict.items():
        f1_score_sum=f1_score_sum+f1_score
    f1_score=f1_score_sum/float(num_classes)
    return f1_score

small_value=0.00001
def compute_f1(TP,FP,FN,compute_type):
    """
    compute f1
    :param TP_micro: number.e.g. 200
    :param FP_micro: number.e.g. 200
    :param FN_micro: number.e.g. 200
    :return: f1_score: a scalar
    """
    precison=TP/(TP+FP+small_value)
    recall=TP/(TP+FN+small_value)
    f1_score=(2*precison*recall)/(precison+recall+small_value)

    if random.choice([x for x in range(500)]) == 1:print(compute_type,"precison:",str(precison),";recall:",str(recall),";f1_score:",f1_score)

    return f1_score
def init_label_dict(num_classes):
    """
    init label dict. this dict will be used to save TP,FP,FN
    :param num_classes:
    :return: label_dict: a dict. {label_index:(0,0,0)}
    """
    label_dict={}
    for i in range(num_classes):
        label_dict[i]=(0,0,0)
    return label_dict

def get_target_label_short(eval_y):
    eval_y_short=[] #will be like:[22,642,1391]
    for index,label in enumerate(eval_y):
        if label>0:
            eval_y_short.append(index)
    return eval_y_short

#get top5 predicted labels
def get_label_using_logits(logits,top_number=5):
    index_list=np.argsort(logits)[-top_number:]
    index_list=index_list[::-1]
    return index_list

#统计预测的准确率
def calculate_accuracy(labels_predicted, labels,eval_counter):
    label_nozero=[]
    #print("labels:",labels)
    labels=list(labels)
    for index,label in enumerate(labels):
        if label>0:
            label_nozero.append(index)
    if eval_counter<2:
        print("labels_predicted:",labels_predicted," ;labels_nozero:",label_nozero)
    count = 0
    label_dict = {x: x for x in label_nozero}
    for label_predict in labels_predicted:
        flag = label_dict.get(label_predict, None)
    if flag is not None:
        count = count + 1
    return count / len(labels)

##################################################

def assign_pretrained_word_embedding(sess,textCNN,id2vec_dict):
#    print("using pre-trained word emebedding.started.word2vec_model_path:",word2vec_model_path)
#    word2vec_model = word2vec.load(word2vec_model_path, kind='bin')
#    word2vec_dict = {}
#    for word, vector in zip(word2vec_model.vocab, word2vec_model.vectors):
#        word2vec_dict[word] = vector
#    word_embedding_2dlist = [[]] * vocab_size  # create an empty word_embedding list.
#    word_embedding_2dlist[0] = np.zeros(FLAGS.embed_size)  # assign empty for first word:'PAD'
#    bound = np.sqrt(6.0) / np.sqrt(vocab_size)  # bound for random variables.
#    count_exist = 0;
#    count_not_exist = 0
#    for i in range(2, vocab_size):  # loop each word. notice that the first two words are pad and unknown token
#        word = vocabulary_index2word[i]  # get a word
#        embedding = None
#        try:
#            embedding = word2vec_dict[word]  # try to get vector:it is an array.
#        except Exception:
#            embedding = None
#        if embedding is not None:  # the 'word' exist a embedding
#            word_embedding_2dlist[i] = embedding;
#            count_exist = count_exist + 1  # assign array to this word.
#        else:  # no embedding for this word
#            word_embedding_2dlist[i] = np.random.uniform(-bound, bound, FLAGS.embed_size);
#            count_not_exist = count_not_exist + 1  # init a random value for the word.
#    word_embedding_final = np.array(word_embedding_2dlist)  # covert to 2d array.
    word_embedding_final = []
    for key,vec in id2vec_dict.items():
        word_embedding_final.append(vec)
    word_embedding_final = np.array(word_embedding_final,dtype=np.float32)
    word_embedding = tf.constant(word_embedding_final, dtype=tf.float32)  # convert to tensor 
    t_assign_embedding = tf.assign(textCNN.Embedding,word_embedding)  # assign this value to our embedding variables of our model.
    sess.run(t_assign_embedding);
#    print("word. exists embedding:", count_exist, " ;word not exist embedding:", count_not_exist)
    print("using pre-trained word emebedding.ended...")

