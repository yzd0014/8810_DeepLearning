
# coding: utf-8

# In[1]:


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

tf.logging.set_verbosity(tf.logging.ERROR)

num_of_samples = 40
sampled_epoch = []
for i in range(num_of_samples):
    sampled_epoch.append(100 + int(700/num_of_samples)*i)
sampled_iterations = []
sampled_epoch_i = 0

x_train = np.expand_dims(np.linspace(0.1,1,200),1)
y_train = np.sin(5*3.14*x_train)/(5*3.14*x_train)
x = tf.placeholder(tf.float32, [None,1], name='x_value')
y = tf.placeholder(tf.float32, [None,1], name='y_value')
#print(y_train)

plt.figure(1)
plt.plot(x_train, y_train, label='training data')

#model 1
h1 = tf.layers.dense(inputs=x, units=10, activation=tf.nn.tanh)   
h2 = tf.layers.dense(inputs=h1, units=18, activation=tf.nn.tanh)        
h3 = tf.layers.dense(inputs=h2, units=15, activation=tf.nn.tanh)
h4 = tf.layers.dense(inputs=h3, units=4, activation=tf.nn.tanh)
output = tf.layers.dense(inputs=h4, units=1, activation=tf.nn.tanh)
loss = tf.losses.mean_squared_error(y, output)

optimizer = tf.train.AdamOptimizer(learning_rate= 0.001).minimize(loss)
grads = tf.train.AdamOptimizer(learning_rate= 0.001).compute_gradients(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

current_iteration = 0
loss_list = []
grad_norm_list = []
global grads_val
batch_size = 32
for epoch in range(1000):
    epoch_loss = 0
    i = 0
    num_of_batches = 0
    while i < len(x_train):
        start = i
        end = i+batch_size
        batch_x = np.array(x_train[start:end])
        batch_y = np.array(y_train[start:end])
        _, loss_val, grads_val = sess.run([optimizer, loss, grads], feed_dict = {x:x_train, y:y_train})
        
        if epoch == sampled_epoch[sampled_epoch_i] and i == 0:
            sampled_iterations.append(current_iteration)
            if sampled_epoch_i < num_of_samples - 1:
                sampled_epoch_i += 1
        
        loss_list.append(loss_val) #record loss 
        i += batch_size
        num_of_batches += 1
        current_iteration += 1
     
    grads_flatten = np.array([])
    for outer_i in range(len(grads_val)):
        grads_flatten = np.append(grads_flatten, grads_val[outer_i][0].flatten())
    grad_norm_list.append(np.linalg.norm(grads_flatten))
    
x_test_1 = np.expand_dims(np.linspace(0.1,1,150),1)
y_predict_1 = sess.run(output, feed_dict={x:x_test_1})
plt.plot(x_test_1, y_predict_1, label='model1')
plt.legend()

#compute minimal ratio
minimal_ratio = [None]*num_of_samples
sampled_loss = [None]*num_of_samples
for i in range(num_of_samples):
    counter = 0
    for j in range(sampled_iterations[i]-50, sampled_iterations[i]):
        if loss_list[j] > loss_list[sampled_iterations[i]]:
            counter = counter + 1
    for j in range(sampled_iterations[i]+1, sampled_iterations[i]+51):
        if loss_list[j] > loss_list[sampled_iterations[i]]:
            counter = counter + 1
    sampled_loss[i] = loss_list[sampled_iterations[i]]
    minimal_ratio[i] = counter/100        

#print(sampled_loss)
plt.figure(2)
plt.scatter(minimal_ratio, sampled_loss)
plt.ylabel('loss')
plt.xlabel('minimal ratio')
plt.show()

sess.close()

