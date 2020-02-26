
# coding: utf-8

# In[1]:


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
tf.logging.set_verbosity(tf.logging.ERROR)

from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/MNIST/', one_hot=True)

#model A=============================================================
x = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, 10])
A_y_true_cls = tf.argmax(y, dimension=1)

A_h1 = {'weights':tf.Variable(tf.random_normal([784, 128])),
                    'biases':tf.Variable(tf.random_normal([128]))}

A_h2 = {'weights':tf.Variable(tf.random_normal([128, 128])),
                    'biases':tf.Variable(tf.random_normal([128]))}

A_output_layer = {'weights':tf.Variable(tf.random_normal([128, 10])),
                    'biases':tf.Variable(tf.random_normal([10]))}

A_l1 = tf.add(tf.matmul(x, A_h1['weights']), A_h1['biases'])
A_l1 = tf.nn.relu(A_l1)

A_l2 = tf.add(tf.matmul(A_l1, A_h2['weights']), A_h2['biases'])
A_l2 = tf.nn.relu(A_l2)

A_output = tf.add(tf.matmul(A_l2, A_output_layer['weights']), A_output_layer['biases'])

A_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=A_output))
A_optimizer = tf.train.AdamOptimizer().minimize(A_loss)
A_y_pred = tf.nn.softmax(logits=A_output)
A_y_pred_cls = tf.argmax(A_y_pred, dimension = 1)#it's used to compute accuracy
A_correct_prediction = tf.equal(A_y_pred_cls, A_y_true_cls)
A_accuracy = tf.reduce_mean(tf.cast(A_correct_prediction, tf.float32))

#model B================================================================
B_y_true_cls = tf.argmax(y, dimension=1)

B_h1 = {'weights':tf.Variable(tf.random_normal([784, 128])),
                    'biases':tf.Variable(tf.random_normal([128]))}

B_h2 = {'weights':tf.Variable(tf.random_normal([128, 128])),
                    'biases':tf.Variable(tf.random_normal([128]))}

B_output_layer = {'weights':tf.Variable(tf.random_normal([128, 10])),
                    'biases':tf.Variable(tf.random_normal([10]))}

B_l1 = tf.add(tf.matmul(x, B_h1['weights']), B_h1['biases'])
B_l1 = tf.nn.relu(B_l1)

B_l2 = tf.add(tf.matmul(B_l1, B_h2['weights']), B_h2['biases'])
B_l2 = tf.nn.relu(B_l2)

B_output = tf.add(tf.matmul(B_l2, B_output_layer['weights']), B_output_layer['biases'])

B_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=B_output))
B_optimizer = tf.train.AdamOptimizer().minimize(B_loss)
B_y_pred = tf.nn.softmax(logits=B_output)
B_y_pred_cls = tf.argmax(B_y_pred, dimension = 1)#it's used to compute accuracy
B_correct_prediction = tf.equal(B_y_pred_cls, B_y_true_cls)
B_accuracy = tf.reduce_mean(tf.cast(B_correct_prediction, tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

#training process A
loss_list = []
acc_list = []
batch_size = 64


for epoch in range(10):
    epoch_loss = 0
    epoch_accuracy = 0
    num_of_batches = 0
    
    for _ in range(int(data.train.num_examples/batch_size)):
        x_batch, y_batch = data.train.next_batch(batch_size)
        _, loss_val, acc_val = sess.run([A_optimizer, A_loss, A_accuracy], feed_dict = {x:x_batch, y:y_batch})
        epoch_loss += loss_val
        epoch_accuracy += acc_val
        num_of_batches += 1
        #print(num_of_batches)
    
    epoch_loss = epoch_loss/num_of_batches    
    epoch_accuracy = epoch_accuracy/num_of_batches
    loss_list.append(epoch_loss)
    acc_list.append(epoch_accuracy)

#sess.close()
#tf.reset_default_graph()

plt.figure(1)
plt.plot(loss_list)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
#plt.show()

plt.figure(2)
plt.plot(acc_list)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()


# In[2]:


#training process B
loss_list = []
acc_list = []
batch_size = 1024

for epoch in range(10):
    epoch_loss = 0
    epoch_accuracy = 0
    num_of_batches = 0
    
    for _ in range(int(data.train.num_examples/batch_size)):
        x_batch, y_batch = data.train.next_batch(batch_size)
        _, loss_val, acc_val = sess.run([B_optimizer, B_loss, B_accuracy], feed_dict = {x:x_batch, y:y_batch})
        epoch_loss += loss_val
        epoch_accuracy += acc_val
        num_of_batches += 1
        #print(num_of_batches)
    
    epoch_loss = epoch_loss/num_of_batches    
    epoch_accuracy = epoch_accuracy/num_of_batches
    loss_list.append(epoch_loss)
    acc_list.append(epoch_accuracy)

plt.figure(3)
plt.plot(loss_list)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
#plt.show()

plt.figure(4)
plt.plot(acc_list)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()


# In[3]:


A_h1_w = sess.run(A_h1['weights'])
A_h1_b = sess.run(A_h1['biases'])
A_h2_w = sess.run(A_h2['weights'])
A_h2_b = sess.run(A_h2['biases'])
A_o_w = sess.run(A_output_layer['weights'])
A_o_b = sess.run(A_output_layer['biases'])

B_h1_w = sess.run(B_h1['weights'])
B_h1_b = sess.run(B_h1['biases'])
B_h2_w = sess.run(B_h2['weights'])
B_h2_b = sess.run(B_h2['biases'])
B_o_w = sess.run(B_output_layer['weights'])
B_o_b = sess.run(B_output_layer['biases'])

sess.close()
tf.reset_default_graph()

#training mode C
alpha = np.linspace(0.0, 1.0, num=100)
alpha_loss_list_train = []
alpha_loss_list_test = []
alpha_acc_list_train = []
alpha_acc_list_test = []

for local_alpha in alpha:
    
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y = tf.placeholder(tf.float32, shape=[None, 10])
    y_true_cls = tf.argmax(y, dimension=1)
    
    #generating mixed model
    model_mix_h1 = {'weights':tf.Variable(tf.random_normal([784, 128])),
                    'biases':tf.Variable(tf.random_normal([128]))}
    model_mix_h1['weights'] = (1-local_alpha) * A_h1_w + local_alpha * B_h1_w
    model_mix_h1['biases'] = (1-local_alpha) * A_h1_b + local_alpha * B_h1_b
    
    model_mix_h2 = {'weights':tf.Variable(tf.random_normal([128, 128])),
                    'biases':tf.Variable(tf.random_normal([128]))}
    model_mix_h2['weights'] = (1-local_alpha) * A_h2_w + local_alpha * B_h2_w
    model_mix_h2['biases'] = (1-local_alpha) * A_h2_b + local_alpha * B_h2_b

    model_mix_output = {'weights':tf.Variable(tf.random_normal([128, 10])),
                    'biases':tf.Variable(tf.random_normal([10]))}
    model_mix_output['weights'] = (1-local_alpha) * A_o_w + local_alpha * B_o_w
    model_mix_output['biases'] = (1-local_alpha) * A_o_b + local_alpha * B_o_b
    
    l1 = tf.add(tf.matmul(x, model_mix_h1['weights']), model_mix_h1['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, model_mix_h2['weights']), model_mix_h2['biases'])
    l2 = tf.nn.relu(l2)

    output = tf.add(tf.matmul(l2, model_mix_output['weights']), model_mix_output['biases'])

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output))
    y_pred = tf.nn.softmax(logits=output)
    y_pred_cls = tf.argmax(y_pred, dimension = 1)#it's used to compute accuracy
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    #test with train data
    test_loss = 0
    test_acc = 0
    num_of_batches = 0
    for _ in range(int(data.train.num_examples/batch_size)):
        x_test, y_test = data.train.next_batch(batch_size)
        loss_val, acc_val = sess.run([loss, accuracy], feed_dict = {x:x_test, y:y_test})
        test_loss += loss_val
        test_acc += acc_val
        num_of_batches += 1
   
    test_loss = test_loss/num_of_batches
    test_acc = test_acc/num_of_batches
    
    alpha_loss_list_train.append(test_loss)
    alpha_acc_list_train.append(test_acc)
   
    
    #test with test data
    test_loss = 0
    test_acc = 0
    num_of_batches = 0
    for _ in range(int(data.test.num_examples/batch_size)):
        x_test, y_test = data.test.next_batch(batch_size)
        loss_val, acc_val = sess.run([loss, accuracy], feed_dict = {x:x_test, y:y_test})
        test_loss += loss_val
        test_acc += acc_val
        num_of_batches += 1
   
    test_loss = test_loss/num_of_batches
    test_acc = test_acc/num_of_batches
        
    alpha_loss_list_test.append(test_loss)
    alpha_acc_list_test.append(test_acc)
        
    sess.close()
    tf.reset_default_graph()
    
plt.figure(5)
plt.plot(alpha, alpha_loss_list_train, label = 'train')
plt.plot(alpha, alpha_loss_list_test, label = 'test')
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('alpha')
plt.legend()
#plt.show()

plt.figure(6)
plt.plot(alpha, alpha_acc_list_train, label = 'train')
plt.plot(alpha, alpha_acc_list_test, label = 'test')
plt.title('accuracy')
plt.ylabel('accuracy')
plt.xlabel('alpha')
plt.legend()

plt.show()
sess.close()

