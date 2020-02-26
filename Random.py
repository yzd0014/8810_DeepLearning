
# coding: utf-8

# In[1]:


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

tf.logging.set_verbosity(tf.logging.ERROR)

from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/MNIST/', one_hot=True)

#print(data.test.num_examples)
np.random.shuffle(data.train.labels)


# In[2]:


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

#training process
sess = tf.Session()
sess.run(tf.global_variables_initializer())

loss_list = []
acc_list = []
test_loss_list = []

batch_size = 64
for epoch in range(50):
    epoch_loss = 0
    epoch_accuracy = 0
    num_of_batches = 0
    
    for i in range(int(data.train.num_examples/batch_size)):
        x_batch, y_batch = data.train.next_batch(batch_size)
        _, loss_val, acc_val = sess.run([optimizer, loss, accuracy], feed_dict = {x:x_batch, y:y_batch})
        epoch_loss += loss_val
        epoch_accuracy += acc_val
        num_of_batches += 1
        
    epoch_loss = epoch_loss/num_of_batches    
    epoch_accuracy = epoch_accuracy/num_of_batches
    loss_list.append(epoch_loss)
    acc_list.append(epoch_accuracy)
    
    #test
    x_test, y_test = data.test.next_batch(batch_size)
    loss_val = sess.run(loss, feed_dict = {x:x_test, y:y_test})
    test_loss_list.append(loss_val)
    
#print(sess.run(accuracy, feed_dict = {x:data.train.images, y:data.train.labels}))
plt.plot(loss_list, label='train')
plt.plot(test_loss_list, label='test')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.show()
sess.close()

