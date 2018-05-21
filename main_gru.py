from GRUClass import *
# import data
from loader import *
import pdb
import pickle
 
conf={
    'lst_layers':[800,800],
    'n_epochs':100,
    'batch_size':40,
    'learning_rate':5e-3,
    'embedding_size':800,
    'voc_size':8000,
    'l2':0,
    'seed':123,
    'data_path':'...',
    'outpath':'...',
    'optim':'rmsprop',
    'lrdr':0.9,
     #adam
    'b1':0.9,
    'b2':0.99,
     #rmsprop
    'decay_rate':0.9
}

lst_layers = conf['lst_layers']
batch_size = conf['batch_size']
 
x_train, x_valid,x_test, w2idx, idx2w = LoadData(conf['data_path'],vocabulary_size=conf['voc_size'])

sampleSize  =x_train[0].shape[0]
model = GRU(embedding_size = conf['embedding_size'], 
             num_classes=conf['voc_size'], 
             lst_layers=lst_layers,
             ckpt_path=conf['outpath'],
             model_name='hani_gru'+str(np.random.randint(1000)))

batch = tf.Variable(0, dtype='int32',trainable=False)
learning_rate = tf.train.exponential_decay(
      conf['learning_rate'],                # Base learning rate.
      batch * batch_size,           # Current index into the dataset.
      sampleSize,                           # Decay step.
      conf['lrdr'],                # Decay rate.
      staircase=False)

if conf['optim'] =='rmsprop':
    optim = tf.train.RMSPropOptimizer(learning_rate=learning_rate,decay=conf['decay_rate']).minimize(model.loss,global_step=batch)
else:
    optim = tf.train.AdamOptimizer   (learning_rate=learning_rate,beta1=conf['b1'],beta2=conf['b2']).minimize(model.loss,global_step=batch)
   

saver = tf.train.Saver(max_to_keep=3)
validFreq= 1
bValidLoss=np.inf
count=int(x_train[0].shape[0]/batch_size)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_loss = 0
    init=np.zeros([ batch_size, np.sum(lst_layers)])
    try:
        for epoch in range(conf['n_epochs']):
            # pdb.set_trace()
            for i in range(count):
                # pdb.set_trace()
                xs, ys, mask =  x_train[0][i*batch_size:(i+1)*batch_size ,0:-1],  \
                                x_train[0][i*batch_size:(i+1)*batch_size, 1:]  ,  \
                                x_train[1][i*batch_size:(i+1)*batch_size, 1:]
                _, train_loss_ = sess.run([optim, model.loss], feed_dict = {
                        model.xs_ : xs,
                        model.ys_ : ys.flatten(),
                        model.mask_:mask.flatten(),
                        model.init_state : init
                    })
                # pdb.set_trace()
                train_loss += train_loss_
            print('[{}] loss : {}'.format(epoch,train_loss/count))
            print('learning_rate : {}'.format(sess.run(learning_rate)))
            if (1+epoch)%validFreq ==0:
                valid_loss = 0
                for i in range(int(x_valid[0].shape[0]/batch_size)):
                    # pdb.set_trace()
                    xs, ys, mask =  x_valid[0][i*batch_size:(i+1)*batch_size ,0:-1],  \
                                    x_valid[0][i*batch_size:(i+1)*batch_size, 1:]  ,  \
                                    x_valid[1][i*batch_size:(i+1)*batch_size, 1:]
                    _, vl = sess.run([optim, model.vtloss], feed_dict = {
                            model.xs_ : xs,
                            model.ys_ : ys.flatten(),
                            model.mask_:mask.flatten(),
                            model.init_state : init
                        })
                    # pdb.set_trace()
                    valid_loss += vl
                if valid_loss<bValidLoss:
                    bValidLoss=valid_loss
                    print('**** [{}] Valid loss : {}'.format(epoch,valid_loss/x_valid[0].shape[0]))
                    saver.save(sess, model.ckpt_path + model.model_name, global_step=epoch)
                    pickle.dump([conf,[w2idx,idx2w],[bValidLoss,train_loss]],open(conf['outpath']+ model.model_name+'.pkl','wb'))
                    print("-------------------------------->model saved<--------------------------------")    
                else:
                    print('[{}] Valid loss : {}'.format(epoch,valid_loss/x_valid[0].shape[0]))
            train_loss = 0

    except KeyboardInterrupt:
        print('interrupted by user at ' + str(i))
       
    
 
