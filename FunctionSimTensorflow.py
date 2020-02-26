import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

tf.logging.set_verbosity(tf.logging.ERROR)

x_train = np.expand_dims(np.linspace(0.1,1,200),1)
y_train = np.sin(5*3.14*x_train)/(5*3.14*x_train)
x = tf.placeholder(tf.float64, [None,1])
y = tf.placeholder(tf.float64, [None,1])
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

sess = tf.Session()
sess.run(tf.global_variables_initializer())

loss_list_1=[]
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
        _, loss_val = sess.run([optimizer, loss], feed_dict = {x:x_train, y:y_train})
        epoch_loss += loss_val
        i = i + batch_size
        num_of_batches += 1
    epoch_loss = epoch_loss/num_of_batches    
    loss_list_1.append(epoch_loss)
    
x_test_1 = np.expand_dims(np.linspace(0.1,1,150),1)
y_predict_1 = sess.run(output, feed_dict={x:x_test_1})
plt.plot(x_test_1, y_predict_1, label='model1')
sess.close()

#model 2
h1 = tf.layers.dense(inputs=x, units=190, activation=tf.nn.tanh)   
output = tf.layers.dense(inputs=h1, units=1, activation=tf.nn.tanh)
loss = tf.losses.mean_squared_error(y, output)
optimizer = tf.train.AdamOptimizer(learning_rate= 0.001).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

loss_list_2=[]
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
        _, loss_val = sess.run([optimizer, loss], feed_dict = {x:x_train, y:y_train})
        epoch_loss += loss_val
        i = i + batch_size
        num_of_batches += 1
    epoch_loss = epoch_loss/num_of_batches    
    loss_list_2.append(epoch_loss)
    
x_test_2 = np.expand_dims(np.linspace(0.1,1,150),1)
y_predict_2 = sess.run(output, feed_dict={x:x_test_2})
plt.plot(x_test_2, y_predict_2, label='model2')
plt.legend()
plt.show()

plt.figure(2)
plt.plot(loss_list_1, label='model1')
plt.plot(loss_list_2, label='model2')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.show()
sess.close()