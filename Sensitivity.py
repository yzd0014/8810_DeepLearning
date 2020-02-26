
# coding: utf-8

# In[1]:


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

tf.logging.set_verbosity(tf.logging.ERROR)

from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/MNIST/', one_hot=True)

batch_size = 64

batch_size_list = []

train_loss_list = []
test_loss_list = []

train_acc_list = []
test_acc_list = []

sensitivity_list = []

for i in range (5):
    print('training model', i)
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
    grads = tf.train.AdamOptimizer().compute_gradients(loss)

    batch_size_list.append(batch_size)

    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
     
    #training process
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    loss_list = []
    acc_list = []
    global grads_val
    
    for epoch in range(10):
        epoch_loss = 0
        epoch_accuracy = 0
        num_of_batches = 0

        for _ in range(int(data.train.num_examples/batch_size)):
            x_batch, y_batch = data.train.next_batch(batch_size)
            _, grads_val, loss_val, acc_val = sess.run([optimizer, grads, loss, accuracy], feed_dict = {x:x_batch, y:y_batch})
            epoch_loss += loss_val
            epoch_accuracy += acc_val
            num_of_batches += 1

        epoch_loss = epoch_loss/num_of_batches    
        epoch_accuracy = epoch_accuracy/num_of_batches
        loss_list.append(epoch_loss)
        acc_list.append(epoch_accuracy)
        
    grads_flatten = np.array([])
    for outer_i in range(len(grads_val)):
        grads_flatten = np.append(grads_flatten, grads_val[outer_i][0].flatten())
        
    #test
    test_loss = 0
    test_acc = 0
    num_of_batches = 0
    for _ in range(int(data.test.num_examples/64)):
        x_test, y_test = data.test.next_batch(64)
        loss_val, acc_val = sess.run([loss, accuracy], feed_dict = {x:x_test, y:y_test})
        test_loss += loss_val
        test_acc += acc_val
        num_of_batches += 1

    test_loss = test_loss/num_of_batches
    test_acc = test_acc/num_of_batches
    
    sess.close()
    tf.reset_default_graph()
    
    train_loss_list.append(loss_list[-1])
    test_loss_list.append(test_loss)
    
    train_acc_list.append(acc_list[-1])
    test_acc_list.append(test_acc)
    
    sensitivity_list.append(np.linalg.norm(grads_flatten))

    batch_size = batch_size * 2
    print('done training model', i)
    print('\n')
    
plt.figure(1)
plt.plot(batch_size_list, train_loss_list, label='train')
plt.plot(batch_size_list, test_loss_list, label='test')
plt.ylabel('loss')
plt.xlabel('batch size')
#plt.legend()


plt.figure(2)
plt.plot(batch_size_list, train_acc_list, label='train')
plt.plot(batch_size_list, test_acc_list, label='test')
plt.ylabel('accuracy')
plt.xlabel('batch size')
#plt.legend()

plt.figure(3)
plt.plot(batch_size_list, sensitivity_list)
plt.ylabel('sensitivity')
plt.xlabel('batch size')
#plt.legend()

plt.show()

