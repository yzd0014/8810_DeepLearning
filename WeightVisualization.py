
# coding: utf-8

# In[ ]:


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

tf.logging.set_verbosity(tf.logging.ERROR)

from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/MNIST/', one_hot=True)

x = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, 10])
y_true_cls = tf.argmax(y, dimension=1)

#create model
h1 = tf.layers.dense(inputs=x, name='h1', units=128, activation=tf.nn.relu)   
h2 = tf.layers.dense(inputs=h1, name='h2', units=128, activation=tf.nn.relu)        
output = tf.layers.dense(inputs=h2, name='output_layer', units=10, activation = None)
y_pred = tf.nn.softmax(logits=output)
y_pred_cls = tf.argmax(y_pred, dimension = 1)#it's used to compute accuracy
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output))
optimizer = tf.train.AdamOptimizer().minimize(loss)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#Getting the Weights
trainable_var_list = tf.trainable_variables()
def get_weights_variable(layer_name):
    with tf.variable_scope(layer_name, reuse=True):
        variable = tf.get_variable('kernel')

    return variable

#training process
sess = tf.Session()
sess.run(tf.global_variables_initializer())

global h1_weights
global all_weights
loss_list = []
acc_list = []
batch_size = 64


plt.figure(1)
pca = PCA(n_components=2)
fig = plt.figure(figsize = (16,8))

ax_1 = fig.add_subplot(1,2,1) 
ax_1.set_xlabel('Principal Component 1', fontsize = 15)
ax_1.set_ylabel('Principal Component 2', fontsize = 15)
ax_1.set_title('2 component PCA (first hidden layer)', fontsize = 20)

ax_2 = fig.add_subplot(1,2,2) 
ax_2.set_xlabel('Principal Component 1', fontsize = 15)
ax_2.set_ylabel('Principal Component 2', fontsize = 15)
ax_2.set_title('2 component PCA (whole model)', fontsize = 20)

for epoch in range(10):
    epoch_loss = 0
    epoch_accuracy = 0
    num_of_batches = 0
    
    for i in range(int(data.train.num_examples/batch_size)):
        x_batch, y_batch = data.train.next_batch(batch_size)
        _, loss_val, acc_val = sess.run([optimizer, loss, accuracy], feed_dict = {x:x_batch, y:y_batch})
        epoch_loss += loss_val
        epoch_accuracy += acc_val
        num_of_batches += 1
        if i%10 == 0:
            if i == 0:
                h1_weights = sess.run(get_weights_variable(layer_name = 'h1')).flatten()
                
                all_weights = h1_weights
                all_weights = np.append(all_weights, sess.run(get_weights_variable(layer_name = 'h2')).flatten())
                all_weights = np.append(all_weights, sess.run(get_weights_variable(layer_name = 'output_layer')).flatten())
            else:
                h1_weights = np.row_stack((h1_weights, sess.run(get_weights_variable(layer_name = 'h1')).flatten()))
                
                temp_weights = sess.run(get_weights_variable(layer_name = 'h1')).flatten()
                temp_weights = np.append(temp_weights, sess.run(get_weights_variable(layer_name = 'h2')).flatten())
                temp_weights = np.append(temp_weights, sess.run(get_weights_variable(layer_name = 'output_layer')).flatten())
                all_weights = np.row_stack((all_weights, temp_weights))

    epoch_loss = epoch_loss/num_of_batches    
    epoch_accuracy = epoch_accuracy/num_of_batches
    loss_list.append(epoch_loss)
    acc_list.append(epoch_accuracy)
  
    #print(h1_weights)
    principalComponents_h1 = pca.fit_transform(h1_weights)
    ax_1.scatter(principalComponents_h1[:,0], principalComponents_h1[:,1], label = 'training #'+ str(epoch), alpha=0.5)
    principalComponents_output = pca.fit_transform(all_weights)
    ax_2.scatter(principalComponents_output[:,0], principalComponents_output[:,1], label = 'training #'+ str(epoch), alpha=0.5)
ax_1.legend()
ax_2.legend()
plt.show()


plt.figure(1)   
plt.plot(loss_list)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')

plt.figure(2)
plt.plot(acc_list)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()
sess.close()




