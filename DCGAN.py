
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math

tf.reset_default_graph()
tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.keras import datasets
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images = tf.image.resize_images(train_images, [64, 64])


# In[2]:


image_size = 64
image_channels = 3
n_noise = 100

X = tf.placeholder(tf.float32, [None, image_size, image_size, image_channels], name='X')
Z = tf.placeholder(tf.float32, [None, n_noise], name='Z')
Lr = tf.placeholder(tf.float32, [], name='Lr')


# In[3]:


n_W1 = 1024
n_W2 = 512
n_W3 = 256
n_W4 = 128
n_W5 = 64
n_hidden = 4*4*n_W1

#generator
def generator(G_input, reuse = False):
    with tf.variable_scope("generator") as scope:
        if reuse:
            scope.reuse_variables()
        G_FW1 = tf.get_variable('G_FW1', [n_noise, n_hidden], initializer = tf.random_normal_initializer(stddev=0.01))
        G_Fb1 = tf.get_variable('G_Fb1', [n_hidden], initializer = tf.constant_initializer(0))

        G_W1 = tf.get_variable('G_W1', [5,5,n_W2, n_W1], initializer = tf.truncated_normal_initializer(stddev=0.02))
        G_W2 = tf.get_variable('G_W2', [5,5,n_W3, n_W2], initializer = tf.truncated_normal_initializer(stddev=0.02))
        G_W3 = tf.get_variable('G_W3', [5,5,n_W4, n_W3], initializer = tf.truncated_normal_initializer(stddev=0.02))
        G_W4 = tf.get_variable('G_W4', [5,5,image_channels, n_W4], initializer = tf.truncated_normal_initializer(stddev=0.02))

    hidden = tf.nn.relu(tf.matmul(G_input, G_FW1) + G_Fb1)
    hidden = tf.reshape(hidden, [batch_size, 4,4,n_W1]) 
    dconv1 = tf.nn.conv2d_transpose(hidden, G_W1, [batch_size, 8, 8, n_W2], [1, 2, 2, 1])
    dconv1 = tf.nn.relu(tf.contrib.layers.batch_norm(dconv1,decay=0.9, epsilon=1e-5))

    dconv2 = tf.nn.conv2d_transpose(dconv1, G_W2, [batch_size, 16, 16, n_W3], [1, 2, 2, 1])
    dconv2 = tf.nn.relu(tf.contrib.layers.batch_norm(dconv2,decay=0.9, epsilon=1e-5))

    dconv3 = tf.nn.conv2d_transpose(dconv2, G_W3, [batch_size, 32, 32, n_W4], [1, 2, 2, 1])
    dconv3 = tf.nn.relu(tf.contrib.layers.batch_norm(dconv3,decay=0.9, epsilon=1e-5))
    
    dconv4 = tf.nn.conv2d_transpose(dconv3, G_W4, [batch_size, 64, 64, image_channels], [1, 2, 2, 1])
    #dconv3 = tf.nn.conv2d_transpose(dconv2, G_W3, [batch_size, 32, 32, image_channels], [1, 2, 2, 1])

    output = tf.nn.tanh(dconv4)
    return output


# In[4]:


#discriminator
def discriminator(D_input, reuse = False):
    with tf.variable_scope("discriminator") as scope:
        if reuse:
            scope.reuse_variables()
        D_W1 = tf.get_variable('D_W1', [5,5,image_channels, n_W5], initializer = tf.truncated_normal_initializer(stddev=0.02))
        D_W2 = tf.get_variable('D_W2', [5,5,n_W5, n_W4], initializer = tf.truncated_normal_initializer(stddev=0.02))
        D_W3 = tf.get_variable('D_W3', [5,5,n_W4, n_W3], initializer = tf.truncated_normal_initializer(stddev=0.02))
        D_W4 = tf.get_variable('D_W4', [5,5,n_W3, n_W2], initializer = tf.truncated_normal_initializer(stddev=0.02)) 
        
        D_FW1 = tf.get_variable('D_FW1', [4*4*n_W2, 1], initializer = tf.random_normal_initializer(stddev=0.01))
        D_Fb1 = tf.get_variable('D_Fb1', [1], initializer = tf.constant_initializer(0))

    conv1 = tf.nn.conv2d(D_input, D_W1, strides = [1, 2, 2, 1], padding='SAME')
    conv1 = tf.nn.leaky_relu(conv1, alpha = 0.2)

    conv2 = tf.nn.conv2d(conv1, D_W2, strides = [1, 2, 2, 1], padding='SAME')
    conv2 = tf.nn.leaky_relu(tf.contrib.layers.batch_norm(conv2, decay=0.9, epsilon=1e-5), alpha = 0.2)

    conv3 = tf.nn.conv2d(conv2, D_W3, strides = [1, 2, 2, 1], padding='SAME')
    conv3 = tf.nn.leaky_relu(tf.contrib.layers.batch_norm(conv3, decay=0.9, epsilon=1e-5), alpha = 0.2)

    conv4 = tf.nn.conv2d(conv3, D_W4, strides = [1, 2, 2, 1], padding='SAME')
    conv4 = tf.nn.leaky_relu(tf.contrib.layers.batch_norm(conv4, decay=0.9, epsilon=1e-5), alpha = 0.2)
    
    hidden = tf.reshape(conv4, [batch_size, 4*4*n_W2]) 

    output = tf.nn.sigmoid(tf.matmul(hidden, D_FW1) + D_Fb1)
    return output


# In[5]:


batch_size = 32

generator_output = generator(Z)
d_fake = discriminator(generator_output, reuse=False)
d_real = discriminator(X, reuse=True)

d_loss = tf.reduce_mean(tf.log(d_real) + tf.log(1 - d_fake))
g_loss = tf.reduce_mean(tf.log(d_fake))

d_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator')
g_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')

d_optimizer = tf.train.AdamOptimizer(Lr, beta1=0.5).minimize(-d_loss, var_list=d_var_list)
g_optimizer = tf.train.AdamOptimizer(Lr, beta1=0.5).minimize(-g_loss, var_list=g_var_list)


# In[6]:


#image processing
sess = tf.Session()
train_images = sess.run(train_images)
train_images = train_images / (255.0 * 0.5) - 1.0


idx = (train_labels == 1).reshape(train_images.shape[0])
train_images = train_images[idx]

print(train_images.shape)
plt.imshow((train_images[10] + 1.0)*0.5)
plt.show()


# In[7]:


#train
import time

sess.run(tf.global_variables_initializer())
epochs = 40
for epoch in range(epochs):
    #learning_rate = 0.0002 * math.pow(0.2, math.floor((epoch + 1) / 3))
    learning_rate = 0.0002
    batch_i = 0
    start_time = time.time()
    while batch_i < len(train_images):
        start = batch_i
        end = batch_i + batch_size
        if end >= len(train_images):
            break
        batch_X = train_images[start:end]
        batch_Z = np.random.uniform(-1., 1., size=[batch_size, n_noise])
        _, loss_val_D = sess.run([d_optimizer, d_loss], feed_dict={X: batch_X, Z: batch_Z, Lr: learning_rate})
        _, loss_val_G = sess.run([g_optimizer, g_loss], feed_dict={Z: batch_Z, Lr: learning_rate})
        batch_i += batch_size
    
    
    #new_image = sess.run(generator_output, feed_dict={Z: batch_Z})
    print('Epoch {:3} D_Loss: {:>6.3f} G_Loss: {:>6.4f} Epoch duration: {:>6.3f}s'.format(epoch, loss_val_D, loss_val_G, time.time() - start_time))


# In[17]:


#testing
batch_Z = np.random.uniform(-1., 1., size=[batch_size, n_noise])
# import pickle
# pickle.dump(batch_Z, open('testing_data.p', 'wb'))

new_images = sess.run(generator_output, feed_dict={Z: batch_Z})


# In[19]:


for i in range(10):
    new_image = (new_images[i] + 1.0)*0.5
    plt.imshow(new_image)
    plt.show()
    #plt.imsave(str(i)+'.png', new_images[i])

