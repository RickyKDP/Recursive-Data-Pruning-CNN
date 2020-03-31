import pandas as pd
import numpy as np
from CNN.CNN_train import CNN,CNN_load
from Word_Elimination.eval_net import *
import os
import progressbar
from Metric.metric import *
from sklearn import preprocessing
from filter_fine_tuning.fine_tuning import *
from filter_init.filter_init import *
import nltk
from nltk.corpus import stopwords
#from tb_embedding_visualization.tensorboard_embedding import embedding_project
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


############# para tuning #############
num_filter = 96
lr = 0.03
single_val_percent = 0.6
LDA_val_percent = 0.85
tuning_percent_list = [0.5,0.7]
tuning_epoch_list = [30,50]
new_filter_percent = 0.5
filter_percent = 0.85
drop_value_percent = 0.5

#########################################

def unique(list_input):
    new_list = [list_input[0]]
    for item in list_input:
        if item in new_list:
            continue
        else:
            new_list.append(item)
    return new_list

def sent_prune(sent,keep_list):
    new_sent = []
    for i,word in enumerate(sent):
        if i in keep_list:
            new_sent.append(word)
    return new_sent

def one_hot2vec(matrix):
    vec = []
    for item in matrix:
        for index,item_ in enumerate(item):
            if item_ == 1:
                vec.append(index)
                break
    return vec

def words_stat(pos_vec,samples,ix2word,length):
    words_stat_dict = {}
    for sample_index,start_ix in enumerate(pos_vec):
        sample = [0]*500
        for index,value in enumerate(samples[sample_index]):
            sample[index] = value
        words = ""
        for _ in range(length):
            words += (ix2word[sample[int(start_ix+_)]] + " ")
        words = words[:-1]
        if words in words_stat_dict.keys():
            words_stat_dict[words] += 1
        else:
            words_stat_dict[words] = 1
    return words_stat_dict

def subset_generate(mat_keep,mat_cond,index):
    subset = mat_cond[:,index:index+1]
    return np.concatenate([mat_keep,subset],axis=1)

########### Load Dataset ##############
test_acc = []
tuning_ix = 0
tmp = 1

filter_num = [num_filter,num_filter,num_filter]
round_num = 0
data = pd.read_pickle("./dataset/ag_news.p")

#label_trn = [i-1 for i in data[3]]
#label_test = [i-1 for i in data[4]]
#data[3] = label_trn
#data[4] = label_test

wd2ix = data[6]

#vocab_emb = data[4]
#vocab_emb = np.array(vocab_emb,dtype=np.float32)

vocab_emb = pd.read_pickle("./dataset/ag_news_glove.p")
vocab_emb = np.array(vocab_emb,dtype=np.float32)


################ Get Convolutional filters ########################
data =[data[0],data[1],data[2],data[3],data[4],data[5]]
#data = pd.read_pickle("./dataset/yelp_full.pkl")


data,Convolution_value,Position_value,classifier_w,conv_filter,CNN_config,test_acc = CNN(data,vocab_emb,filter_num,f"train_{round_num}",[None,None,None],lr,round_num,tuning_ix)

#pd.to_pickle([data,Convolution_value,Position_value,classifier_w,conv_filter,CNN_config,test_acc],f"./yelp_base_CNN.p")

#data,Convolution_value,Position_value,classifier_w,conv_filter,CNN_config,test_acc = pd.read_pickle("yelp_base_CNN.p")

pos_trn = Position_value[0]
trn_data = data[0]

data_left = data[0]

stop_words = set(stopwords.words('english'))

stop_words_ix = []

for wd in stop_words:
    if wd in wd2ix.keys():
        stop_words_ix.append(wd2ix[wd])

for ix,sent in enumerate(data[0]):
    new_sent =[]
    for wd in sent:
        if wd in stop_words_ix:
            new_sent.append(0)
        else:
            new_sent.append(wd)
    data[0][ix] = new_sent

padding_sum = 0
for counter,sent in enumerate(data[0]):
    for word in sent:
        if word == 0:
            padding_sum += 1

print(f"Removed words no. : {padding_sum/(counter+1)}")


################Classifier Layer Weight Analysis########################
for _ in range(20):
    W1,W2 = classifier_w
    W2 = np.absolute(W2)
    W1 = np.absolute(W1)
    W2_sum = np.expand_dims((np.sum(W2,axis=1)),axis=0).T
    filter_score = np.matmul(W1,W2_sum)
    filter_score = np.squeeze(filter_score)
    trn_sample = data[0]

    useful_filter_ix = []
    useless_filter_ix = []
    conv_trn = Convolution_value[0]
    conv_eval = Convolution_value[1]
    pos_trn = Position_value[0]
    pos_eval = Position_value[1]
    eval_score = []


    p = progressbar.ProgressBar()
    keep_conv_set_trn = np.array([[]]*len(conv_trn))
    keep_conv_set_eval = np.array([[]]*len(conv_eval))
    for iter_num in p(range(int(filter_score.shape[0]*0.65))):
        index = np.argmax(filter_score)
        keep_conv_set_trn = np.concatenate([keep_conv_set_trn,np.expand_dims(conv_trn[:,int(index)],axis=1)],axis=1)
        keep_conv_set_eval = np.concatenate([keep_conv_set_eval,np.expand_dims(conv_eval[:,int(index)],axis=1)],axis=1)
        useful_filter_ix.append(index)
#        eval_score.append(filter_set_eval(CNN_config,keep_conv_set_trn,data[3],keep_conv_set_eval,data[4]))
        filter_score[index] = -1000.0
#        print(f"{iter_num}/{len(filter_score)}")

########################################################################

############# dataset cleaning ################
    pos_trn_t = pos_trn.T
    conv_trn_t = conv_trn.T

    useless_filter_ix = []
    for i in range(filter_score.shape[0]):
        if i not in useful_filter_ix:
            useless_filter_ix.append(i)

    pos_drop = [[] for _ in range(len(trn_sample))]
    pos_keep = [[] for _ in range(len(trn_sample))]
    for filter_ix in useful_filter_ix:
        words_len = filter_ix//96+1
        conv_vec = conv_trn_t[filter_ix]
        for ix,pos in enumerate(pos_trn_t[filter_ix]):
            if conv_trn_t[filter_ix][ix] == 0:
                for i in range(words_len):
                    if (pos+i) < len(trn_sample[ix]) and (pos+i) not in pos_drop[ix]:
                        pos_drop[ix].append(pos+i)
            else:
                for i in range(words_len):
                    if (pos+i) < len(trn_sample[ix]) and (pos+i) not in pos_keep[ix]:
                        pos_keep[ix].append(pos+i)

    for filter_ix in useless_filter_ix:
        words_len = filter_ix//96+1
        for ix,pos in enumerate(pos_trn_t[filter_ix]):
            for i in range(words_len):
                if (pos+i) < len(trn_sample[ix]) and (pos+i) not in pos_drop[ix]:
                    pos_drop[ix].append(pos+i)

    for ix,item in enumerate(pos_drop):
        new_drop_list = []
        for word_ix in item:
            if word_ix not in pos_keep[ix]:
                new_drop_list.append(word_ix)
        pos_drop[ix] = new_drop_list


    for sample_ix,drop_list in enumerate(pos_drop):
        for ix in drop_list:
            trn_sample[sample_ix][int(ix)] = 0

    data[0] = trn_sample

    padding_sum = 0
    for counter,sent in enumerate(data[0]):
        for word in sent:
            if word == 0:
                padding_sum += 1

    print(f"Removed words no. : {padding_sum/(counter+1)}")
#################################################
    filter_initial_ = [[],[],[]]
    b_initial_ = [[],[],[]]
    for filter_ix in useful_filter_ix:
        filter_len = filter_ix//96+1
        filter_index = filter_ix%96
        filter_initial_[filter_len-1].append(conv_filter[f"filter_{filter_len}"][0][:,:,:,filter_index:filter_index+1])
        b_initial_[filter_len-1].append(conv_filter[f"filter_{filter_len}"][1][filter_index:filter_index+1])

    filter_initial = []
    b_initial = []
    for filter_mat in filter_initial_:
        shape_one = filter_mat[0].shape
        filter_ = np.concatenate(filter_mat,axis=3)
        while filter_.shape[3] < 96:
            filter_ = np.concatenate([filter_,np.random.uniform(-1,1,shape_one)],axis=3)
        filter_initial.append(filter_)

    for b_mat in b_initial_:
        shape_one = b_mat[0].shape
        b_ = np.concatenate(b_mat,axis=0)
        while b_.shape[0] < 96:
            b_ = np.concatenate([b_,np.random.uniform(-1,1,shape_one)],axis=0)
        b_initial.append(b_)

    preset_para = [filter_initial,b_initial]

    pd.to_pickle([data,useful_filter_ix,pos_drop,preset_para,Convolution_value,Position_value],f"./analyze/back_up_0.5__{round_num}.pkl")

    round_num += 1
#    vocab_emb = conv_filter["embedding"]
    filter_num = [96,96,96]
    data,Convolution_value,Position_value, classifier_w,conv_filter,CNN_config,test_para = CNN(data,vocab_emb,filter_num,f"train_{round_num}",preset_para,lr,round_num,tuning_ix)
    pd.to_pickle([data,Convolution_value,Position_value,classifier_w,conv_filter,CNN_config,test_para],f"ag_news_{round_num}.p")
    pd.to_pickle(test_para,f"./analyze/ag_news/test_para_0.85_{round_num}.pkl")
    tuning_ix += 1




