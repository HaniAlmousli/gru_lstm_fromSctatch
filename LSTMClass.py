import tensorflow as tf
import numpy as np
import pdb
import random
import argparse
import sys
 
 
 
class LSTM():
 
    def __init__(self, embedding_size, num_classes, lst_layers,
            ckpt_path='~/',
            model_name='hani_lstm'):
 
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.lst_layers = lst_layers
        self.ckpt_path = ckpt_path
        self.model_name = model_name
        self.num_layers = len(lst_layers)
        # build graph ops
        def __graph__():
            tf.reset_default_graph()
            # inputs
            xs_ = tf.placeholder(shape=[None, None], dtype=tf.int32)
            ys_ = tf.placeholder(shape=[None], dtype=tf.int32)
            mask_ = tf.placeholder(shape=[None], dtype=tf.int32)
            #
            # embeddings
            embs = tf.get_variable('emb', [num_classes, self.embedding_size])
            rnn_inputs = tf.nn.embedding_lookup(embs, xs_)
            # initializer
            xav_init = tf.contrib.layers.xavier_initializer
            # initial hidden state
            init_state = tf.placeholder(shape=[2, None, np.sum(lst_layers)],
                    dtype=tf.float32, name='initial_state')
 
            # params
            lstW=[];lstU=[];lstb=[]
            inpSize=self.embedding_size
            for i in range(len(lst_layers)):
                lstW.append(tf.get_variable('W'+str(i), shape=[self.lst_layers[i], 4*self.lst_layers[i]], initializer=xav_init()))
                lstU.append(tf.get_variable('U'+str(i), shape=[inpSize, 4*self.lst_layers[i]], initializer=xav_init()))
                lstb.append(tf.get_variable('b'+str(i), shape=[2, self.lst_layers[i]], initializer=tf.constant_initializer(0.)))
                inpSize=self.lst_layers[i]
            ####
            # step - LSTM
            def step(prev, x):
                # gather previous internal state and output state
                st_1, ct_1 = tf.unstack(prev)
                # iterate through layers
                st, ct = [], []
                inp = x
                concat_st = 0
                concat_ct = 0
                for i in range(self.num_layers):
                    # prev_st = tf.slice(st_1,[0,cursor],[-1,cursor+self.lst_layers[i]])
                    # prev_ct = tf.slice(ct_1,[0,cursor],[-1,cursor+self.lst_layers[i]])
                    # cursor += self.lst_layers[i]
 
                    prev_st = tf.slice(st_1,[0,i*self.lst_layers[i]],[-1,self.lst_layers[i]])
                    prev_ct = tf.slice(ct_1,[0,i*self.lst_layers[i]],[-1,self.lst_layers[i]])
 
                    u_out  = tf.matmul(inp    , lstU[i])
                    st_out = tf.matmul(prev_st, lstW[i])
                    g_out = u_out+st_out
                    #  input gate
                    ig = tf.sigmoid(tf.slice(g_out,[0,0],[-1,self.lst_layers[i]]))
                    #  forget gate
                    fg = tf.sigmoid(tf.slice(g_out,[0,self.lst_layers[i]],[-1,self.lst_layers[i]]))
                    #  output gate
                    og = tf.sigmoid(tf.slice(g_out,[0,2*self.lst_layers[i]],[-1,self.lst_layers[i]]))
                    #  gate weights
                    g = tf.tanh(tf.slice(g_out,[0,3*self.lst_layers[i]],[-1,self.lst_layers[i]]))
                    ###
                    # new internal cell state
                    # pdb.set_trace()
                    ct_i = prev_ct*fg + g*ig + lstb[i][0]
                    # output state
                    st_i = tf.tanh(ct_i)*og  + lstb[i][1]
                    inp = st_i
                    if i ==0:
                        concat_st=st_i
                        concat_ct=ct_i
                    else:
                        concat_st = tf.concat([concat_st,st_i],axis=1)
                        concat_ct = tf.concat([concat_ct,ct_i],axis=1)
                return tf.stack([concat_st, concat_ct])
            ###
            #   tf.scan(fn, elems, initializer)
            self.states = tf.scan(step,
                    tf.transpose(rnn_inputs, [1,0,2]),
                    initializer=init_state) #([Dimension(None) {TIME}, Dimension(2) {st,ct}, Dimension(None) {bs}, Dimension(700) {ouput}])
            #
            # predictions
            V  = tf.get_variable('V', shape=[self.lst_layers[-1], num_classes],
                                initializer=xav_init())
            bo = tf.get_variable('bo', shape=[num_classes],
                                 initializer=tf.constant_initializer(0.))
 
            ####
            # get last state before reshape/transpose
            self.last_state = self.states[-1]
            # pdb.set_trace()
            ####
            # pick st
            self.layersoutput = tf.transpose(self.states, [1,2,0,3])[0]
            self.states_reshaped = tf.reshape(self.layersoutput, [-1, np.sum(self.lst_layers)])
 
            last_layer_index=0
            if len(self.lst_layers)>1:
                last_layer_index = np.sum(self.lst_layers[0:-1])
 
            self.last_layer_out  = tf.slice(self.states_reshaped,[0,last_layer_index],[-1,self.lst_layers[-1]])
            self.logits = tf.matmul(self.last_layer_out, V) + bo
            # predictions
            self.predictions = tf.nn.softmax(self.logits)
            #
            # optimization
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,labels= ys_)
            
            loss = tf.reduce_mean(tf.boolean_mask(losses,mask_))
            vtloss = tf.reduce_sum(tf.boolean_mask(losses,mask_))
            self.xs_ = xs_
            self.ys_ = ys_
            self.mask_=mask_
            self.loss = loss
            self.vtloss=vtloss
            self.init_state = init_state
            self.embs=embs
        #####
        # build graph
        sys.stdout.write('\n<log> Building Graph...')
        __graph__()
        sys.stdout.write('</log>\n')

    def GenerateSentence(self,w2idx,idx2w,sentencesCount=10):
        
        #
        # start session
        lst=[]
        with tf.Session() as sess:
            # init session
            sess.run(tf.global_variables_initializer())
            # restore session
            ckpt = tf.train.get_checkpoint_state(self.ckpt_path)
            saver = tf.train.Saver()

            # if ckpt and ckpt.model_checkpoint_path:
            print("RESTORED")
            saver.restore(sess, ckpt.model_checkpoint_path)
            for i in range(sentencesCount):
                # generate operation
                current_word = w2idx["SENTENCE_START"]
                words = [current_word]
                state = None
                # enter the loop
                while words[-1]!=w2idx["SENTENCE_END"]:
                    if state:
                        feed_dict = {self.xs_ : np.array([current_word]).reshape([1,1]),
                                self.init_state : state_}
                    else:
                        feed_dict = {self.xs_ : np.array([current_word]).reshape([1,1]),
                                self.init_state : np.zeros([2, 1, np.sum(self.lst_layers)])}
                    #
                    # forward propagation
                    preds, state_ = sess.run([self.predictions, self.last_state], feed_dict=feed_dict)
                    # 
                    # set flag to true
                    state = True
                    # 
                    # set new word
                    preds=np.squeeze(preds)
                    unknownprob = preds[w2idx['UNKNOWN_TOKEN']]
                    preds=preds+(unknownprob/(len(preds)-1))
                    preds[w2idx['UNKNOWN_TOKEN']]=0
                    current_word = np.random.choice(preds.shape[-1], 1, p=preds)[0]
                    # add to list of words
                    words.append(current_word)
                lst.append(' '.join([idx2w[w] for w in words]))
        return lst
    
