import pickle
import pdb
import itertools
import operator
import numpy as np
import nltk
import sys
import os
import time
from datetime import datetime
import io


def LoadData(data_path = '~/',
             vocabulary_size = 8000,
             minlen = 10,
             maxlen=120):
    rng = np.random.RandomState(123)
    def GetText():
        #'/home/hani/Data/RBC/'
        lst_files = os.listdir(data_path)
        for file in lst_files:
            f = io.open(data_path+file, 'rU', encoding='utf-8')
            txt = f.read()
            txt = txt.replace("\n",".\n")
            yield txt  


    unknown_token = "UNKNOWN_TOKEN"
    sentence_start_token = "SENTENCE_START"
    sentence_end_token = "SENTENCE_END"
    sentences = itertools.chain(*[nltk.sent_tokenize(x.lower()) for x in GetText()])
    sentences = ["%s %s %s" % (sentence_start_token, x.lower(), sentence_end_token) for x in sentences]       
    tokenized_sentences=([nltk.word_tokenize(sent) for sent in sentences])
    #pdb.set_trace()
    #tokenized_sentences = sorted(tokenized_sentences, key=lambda word: word[1])
    word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    vocab = word_freq.most_common(vocabulary_size-1)
    index_to_word = [x[0] for x in vocab]
    index_to_word.append(unknown_token)
    
    
    word_to_index={}    
    for i in range(len(index_to_word)):
        word_to_index[index_to_word[i]]=i
    # pdb.set_trace()    
    tokenized_sentences_Numbers=[]
    for sent in tokenized_sentences:
        if len(sent)>minlen and len(sent)<=maxlen:
            tokenized_sentences_Numbers.append([ word_to_index[w] if w in word_to_index else word_to_index[unknown_token] for w in sent])
    

    data = word_to_index[sentence_end_token]* \
            np.ones([len(tokenized_sentences_Numbers),maxlen],dtype='int32')
    mask = np.zeros([data.shape[0],maxlen],dtype='int32')

    for i in range(data.shape[0]):
        data[i,0:len(tokenized_sentences_Numbers[i])] = tokenized_sentences_Numbers[i]
        mask[i,0:len(tokenized_sentences_Numbers[i])] = 1

    sampleSize= data.shape[0]
    indices = np.arange(sampleSize)
    rng.shuffle(indices)
    r1= int(np.round(0.7 *sampleSize))
    r2= int(np.round(0.85*sampleSize))
    # pdb.set_trace()
    # print(len(indices[0:r1]),"   ",len(indices[r1:r2]))
    return([ (data[indices[0:r1],:]  ,mask[indices[0:r1],:]),
             (data[indices[r1:r2],:] ,mask[indices[r1:r2],:]),
             (data[indices[r2:],:]   ,mask[indices[r2:],:]) ,word_to_index,index_to_word])



def LoadDataAsList(data_path = '~/',
             vocabulary_size = 8000,
             minlen = 30,
             maxlen=120):
    def GetText():
        #'/home/hani/Data/RBC/'
        lst_files = os.listdir(data_path)
        lst_files[0:3000]
        for file in lst_files:
            f = io.open(data_path+file, 'rU', encoding='utf-8')
            txt = f.read()
            txt = txt.replace("\n",".\n")
            yield txt  


    unknown_token = "UNKNOWN_TOKEN"
    sentence_start_token = "SENTENCE_START"
    sentence_end_token = "SENTENCE_END"
    sentences = itertools.chain(*[nltk.sent_tokenize(x.lower()) for x in GetText()])
    sentences = ["%s %s %s" % (sentence_start_token, x.lower(), sentence_end_token) for x in sentences]       
    tokenized_sentences=([nltk.word_tokenize(sent) for sent in sentences])
    #pdb.set_trace()
    #tokenized_sentences = sorted(tokenized_sentences, key=lambda word: word[1])
    word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    vocab = word_freq.most_common(vocabulary_size-1)
    index_to_word = [x[0] for x in vocab]
    index_to_word.append(unknown_token)
    
    
    word_to_index={}    
    for i in range(len(index_to_word)):
        word_to_index[index_to_word[i]]=i
    # pdb.set_trace()    
    tokenized_sentences_Numbers=[]
    for sent in tokenized_sentences:
        sent.append(word_to_index['SENTENCE_END'])
        if len(sent)>minlen and len(sent)<=maxlen:
            tokenized_sentences_Numbers.append([ word_to_index[w] if w in word_to_index else word_to_index[unknown_token] for w in sent])
    

    return tokenized_sentences_Numbers,word_to_index,index_to_word

def rand_batch_gen(tokenized_sentences_Numbers,rng,batchSize=256,lstlen=5500,minlen=30):
    while True:
        batch=np.zeros([batchSize,minlen])
        indices = rng.randint(0,lstlen,batchSize)
        counter=0
        for i in indices:
            s = rng.randint(0,len(tokenized_sentences_Numbers[i])-minlen)
            batch[counter]=tokenized_sentences_Numbers[i][s:(s+30)]
            counter+=0
        yield batch[:,0:-1],batch[:,1:]


