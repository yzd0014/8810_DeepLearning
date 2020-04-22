
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

from scipy.linalg import sqrtm

tf.reset_default_graph()
tf.logging.set_verbosity(tf.logging.ERROR)
all_images = pickle.load(open('all_wgan_images.p', 'rb'))

all_images = all_images[:32000,:,:,:]
print(all_images.shape)


# In[2]:


all_images = all_images * 255
all_images = all_images.astype('int32')
sess = tf.Session()
all_images = tf.image.resize_images(all_images, [299, 299])
all_images = sess.run(all_images)
print(all_images.shape)


# In[3]:


import math
def calculate_IS(images, n_split=10, eps=1E-16):
    model = tf.keras.applications.InceptionV3()

    scores = list()
    n_part = math.floor(images.shape[0] / n_split)
    for i in range(n_split):
        ix_start, ix_end = i * n_part, (i + 1) * n_part
        subset = images[ix_start:ix_end]
        subset = subset.astype('float32')
        subset = tf.keras.applications.inception_v3.preprocess_input(subset)

        p_yx = model.predict(subset)
        print(p_yx.shape)
        # p_yx = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        p_y = np.expand_dims(p_yx.mean(axis=0), 0)
        kl_d = p_yx * (np.log(p_yx + eps) - np.log(p_y + eps))

        sum_kl_d = kl_d.sum(axis=1)
        avg_kl_d = np.mean(sum_kl_d)
        is_score = np.exp(avg_kl_d)
        scores.append(is_score)

    is_avg, is_std = np.mean(scores), np.std(scores)
    return is_avg, is_std


# In[4]:


is_avg, is_std = calculate_IS(all_images)
print('WGAN score', is_avg, is_std)

