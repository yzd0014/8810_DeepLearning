
# coding: utf-8

# In[1]:


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pickle
import copy
import json
import time

tf.logging.set_verbosity(tf.logging.ERROR)

#data processing
with open('MLDS_hw2_1_data/training_label.json') as f:
    training_data = json.load(f)


# In[2]:


#generate mapping table and prepare traning data
x_train = []
y_train = []

text_block = ''
max_sentence_length = -1
for i_video in range(len(training_data)): #len(training_data)
    for i_caption in range(len(training_data[i_video]['caption'])):
        sentence = training_data[i_video]['caption'][i_caption].lower()
        sentence = sentence.replace('.', ' ')
        sentence = sentence.replace('!', ' ')
        text_block = text_block + sentence + ' '
        
        sentence = sentence.split()
        y_train.append(sentence)
        if(len(sentence) > max_sentence_length):
            max_sentence_length = len(sentence)
            
        video_name = training_data[i_video]['id']
        features = np.load('MLDS_hw2_1_data/training_data/feat/'+ video_name + '.npy')
        x_train.append(features)

max_sentence_length = max_sentence_length + 1
x_train = np.array(x_train)
x_train = np.flip(x_train, 1)


# In[3]:


CODES = {'<PAD>': 0, '<EOS>': 1, '<GO>': 2 }
vocab = set(text_block.split())
vocab_2_int = copy.copy(CODES)

for v_i, v in enumerate(vocab, len(CODES)):
    vocab_2_int[v] = v_i
int_2_vocab = {v_i: v for v, v_i in vocab_2_int.items()}

#pad caption
temp_y_train = []
for i in range(len(y_train)):
    s = [vocab_2_int['<GO>']]
    for j in range(len(y_train[i])):
        s.append(vocab_2_int[y_train[i][j]])
    s.append(vocab_2_int['<EOS>'])
    for k in range(len(s), max_sentence_length+1):
        s.append(vocab_2_int['<PAD>'])
    
    s = np.array(s)
    temp_y_train.append(s)

y_train = temp_y_train
y_train = np.array(y_train)

pickle.dump((vocab_2_int, int_2_vocab), open('data_mapping.p', 'wb'))


# In[4]:


print(max_sentence_length)
print(len(y_train))


# In[5]:


embed_size = 10
batch_size = 10 

tf.reset_default_graph()
sess = tf.InteractiveSession()

frame_inputs = tf.placeholder(tf.float32, shape=[None, 80, 4096], name='frame_inputs')
caption_inputs = tf.placeholder(tf.int32, shape=[None, max_sentence_length], name='caption_inputs')
targets = tf.placeholder(tf.int32, shape=[None, max_sentence_length], name='targets')

# decoder inputs embeding
caption_inputs_embeding = tf.Variable(tf.random_uniform((len(vocab_2_int), embed_size), -1.0, 1.0), name='caption_embedding')
caption_inputs_padding = tf.placeholder(dtype=tf.int32, shape=[None, 80], name='caption_inputs_padding')
caption_inputs_padded = tf.concat([caption_inputs_padding, caption_inputs], 1)
caption_inputs_embedded = tf.nn.embedding_lookup(caption_inputs_embeding, caption_inputs_padded)

with tf.variable_scope("encoding") as encoding_scope:
    layer_1_inputs_padding = tf.placeholder(dtype=tf.float32, shape=[None, max_sentence_length, 4096], name='layer_1_inputs_padding')
    layer_1_inputs = tf.concat([frame_inputs, layer_1_inputs_padding], 1)
    
    layer_1_lstm = tf.contrib.rnn.BasicLSTMCell(256)
    layer_1_outputs, _ = tf.nn.dynamic_rnn(layer_1_lstm, inputs=layer_1_inputs, dtype=tf.float32)
    
    
with tf.variable_scope("decoding") as decoding_scope:     
    layer_2_inputs = tf.concat([layer_1_outputs, caption_inputs_embedded], 2)
    
    layer_2_lstm = tf.contrib.rnn.BasicLSTMCell(256)
    layer_2_outputs, _ = tf.nn.dynamic_rnn(layer_2_lstm, inputs=layer_2_inputs, dtype=tf.float32)
    layer_2_outputs_clamp = layer_2_outputs[:,80:80+max_sentence_length,:]
    
#connect outputs to 
logits = tf.contrib.layers.fully_connected(layer_2_outputs_clamp, num_outputs=len(vocab_2_int), activation_fn=None)
logits_proxy = tf.identity(logits, name = 'logits')
with tf.name_scope("optimization"):
    # Loss function
    loss = tf.contrib.seq2seq.sequence_loss(logits_proxy, targets, tf.ones([batch_size, max_sentence_length]))
    # Optimizer
    optimizer = tf.train.RMSPropOptimizer(1e-3).minimize(loss)


# In[7]:


sess.run(tf.global_variables_initializer())
#batch_size = 10 
epochs = 30
for epoch_i in range(epochs):
    batch_i = 0
    start_time = time.time()
    while batch_i < len(x_train):
        start = batch_i
        end = batch_i + batch_size
        if end >= len(x_train):
            break
        source_batch = x_train[start:end]
        target_batch = y_train[start:end]
        caption_padding_batch = np.zeros((batch_size, 80))
        frame_padding_batch = np.zeros((batch_size, max_sentence_length, 4096))
        _, batch_loss, batch_logits = sess.run([optimizer, loss, logits_proxy], 
                        feed_dict={frame_inputs: source_batch, 
                        caption_inputs: target_batch[:,:-1], 
                        targets: target_batch[:,1:],
                        caption_inputs_padding: caption_padding_batch,
                        layer_1_inputs_padding: frame_padding_batch})                           
        batch_i = batch_i + batch_size
     
    accuracy = np.mean(batch_logits.argmax(axis=-1) == target_batch[:,1:])
    print('Epoch {:3} Loss: {:>6.3f} Accuracy: {:>6.4f} Epoch duration: {:>6.3f}s'.format(epoch_i, batch_loss, accuracy, time.time() - start_time))


# In[8]:


saver = tf.train.Saver()
saver.save(sess, 'V2C_Padding') 
print('Model Trained and Saved!') 


# In[9]:


# #testing
# with open('MLDS_hw2_1_data/testing_label.json') as f:
#     testing_data = json.load(f)

# outf = open("test_output.txt", "w")
# batch_size = 1
# for i_video in range(len(testing_data)):
#     video_name = testing_data[i_video]['id']
#     features = np.load('MLDS_hw2_1_data/testing_data/feat/'+ video_name + '.npy')
#     video_feat = [features]
#     video_feat.reverse()

#     predict_caption = np.zeros((batch_size, max_sentence_length)) + vocab_2_int['<PAD>']
#     predict_caption[:,0] = vocab_2_int['<GO>']
    
#     for i in range(max_sentence_length):
#         caption_padding_batch = np.zeros((batch_size, 80))
#         frame_padding_batch = np.zeros((batch_size, max_sentence_length, 4096))
#         test_logits = sess.run(logits_proxy, feed_dict={frame_inputs: video_feat, 
#                                                         caption_inputs: predict_caption,
#                                                         caption_inputs_padding: caption_padding_batch,
#                                                         layer_1_inputs_padding: frame_padding_batch})
#         predict_word = test_logits[:,i].argmax(axis=-1)

#         if predict_word[0] == vocab_2_int['<PAD>']:
#             break
#         if i+1 < max_sentence_length:
#             predict_caption[:,i+1] = predict_word
#         if predict_word[0] == vocab_2_int['<EOS>']: 
#             break
        
#     sentence = ''
#     for word in predict_caption[0]:
#         if word == vocab_2_int['<EOS>'] or word == vocab_2_int['<PAD>']:
#             break
#         if word != vocab_2_int['<GO>']:
#             sentence = sentence + int_2_vocab[word] + ' '

#     sentence = sentence.capitalize()    
#     sentence = list(sentence)
#     sentence[-1] = '.'
#     sentence = ''.join(sentence)
#     sentence = video_name + ',' + sentence
#     #print(sentence)
#     outf.write(sentence)
#     outf.write('\n')

# outf.close()
# print('Finished!')
# #sess.close()
# #tf.reset_default_graph()

