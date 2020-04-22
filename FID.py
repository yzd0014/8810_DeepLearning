
# coding: utf-8

# In[1]:


# example of calculating the frechet inception distance in Keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.linalg import sqrtm

sess = tf.Session()


# In[2]:


from tensorflow.keras import datasets
(train_pictures, train_labels), (test_pictures, test_labels) = datasets.cifar10.load_data()
idx = (train_labels == 1).reshape(train_pictures.shape[0])
train_images = train_pictures[idx]
train_images = train_images[0:10,:,:,:]

train_images = tf.image.resize_images(train_images, [299, 299])
train_images = sess.run(train_images)
train_images = train_images.astype('int32')
print(train_images.shape)

idx = (train_labels == 2).reshape(train_pictures.shape[0])
train_images_bird = train_pictures[idx]
train_images_bird = train_images_bird[0:10,:,:,:]
train_images_bird = tf.image.resize_images(train_images_bird, [299, 299])
train_images_bird = sess.run(train_images_bird)
train_images_bird = train_images.astype('int32')
print(train_images_bird.shape)

train_images2 = np.append(train_images, train_images_bird, axis=0)
print(train_images2.shape)

# idx = (train_labels == 3).reshape(train_pictures.shape[0])
# train_images_cat = train_pictures[idx]
# train_images_cat = train_images_cat[0:10,:,:,:]
# train_images_cat = tf.image.resize_images(train_images_cat, [299, 299])
# train_images_cat = sess.run(train_images_cat)
# train_images_cat = train_images.astype('int32')
# print(train_images_cat.shape)

# train_images3 = np.append(train_images2, train_images_cat, axis=0)
# print(train_images3.shape)




# In[3]:


#load images from dcgan
dcgan_images = []
for i in range(10):
    image =mpimg.imread('dc_images/'+str(i)+'.png')

    image = tf.image.resize_images(image, [299, 299])
    image = sess.run(image)
    
    image = image[:,:,0:3]
    image = image * 255
    image = image.astype('int32')
    
    print(image.shape)
#     plt.imshow(image)
#     plt.show()
    dcgan_images.append(image)
   
dcgan_images = np.array(dcgan_images)
print(dcgan_images.shape)

dcgan_images2 = np.append(dcgan_images, train_images_bird, axis=0)
print(dcgan_images2.shape)


# In[4]:


#load images from wgan
wgan_images = []
for i in range(10):
    image =mpimg.imread('w_images/'+str(i)+'.png')

    image = tf.image.resize_images(image, [299, 299])
    image = sess.run(image)
    
    image = image[:,:,0:3]
    image = image * 255
    image = image.astype('int32')
    
    print(image.shape)
#     plt.imshow(image)
#     plt.show()
    wgan_images.append(image)
   
wgan_images = np.array(wgan_images)
print(wgan_images.shape)

wgan_images2 = np.append(wgan_images, train_images_bird, axis=0)
print(wgan_images2.shape)


# In[ ]:


def calculate_fid(model, images1, images2):
    act1 = model.predict(images1)
    act2 = model.predict(images2)
    
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    
    ssdiff = np.sum((mu1-mu2)**2.0)
    
    covmean = sqrtm(sigma1.dot(sigma2))
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid
    


# In[ ]:


model = tf.keras.applications.InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))

dcgan_images = dcgan_images.astype('float32')
wgan_images = wgan_images.astype('float32')
train_images = train_images.astype('float32')

dcgan_images = tf.keras.applications.inception_v3.preprocess_input(dcgan_images)
wgan_images = tf.keras.applications.inception_v3.preprocess_input(wgan_images)
train_images = tf.keras.applications.inception_v3.preprocess_input(train_images)

dcfid = calculate_fid(model, dcgan_images, train_images)
print('FID (dcgan): %.3f' % dcfid)

wfid = calculate_fid(model, wgan_images, train_images)
print('FID (wgan): %.3f' % wfid)


# In[7]:


# import math
# def calculate_IS(images, n_split=10, eps=1E-16):
#     model = tf.keras.applications.InceptionV3()
    
#     scores = list()
#     n_part = math.floor(images.shape[0]/n_split)
#     for i in range(n_split):
#         ix_start, ix_end = i*n_part, (i+1)*n_part
#         subset = images[ix_start:ix_end]
#         subset = subset.astype('float32')
#         subset = tf.keras.applications.inception_v3.preprocess_input(subset)
        
#         p_yx = model.predict(subset)
#         #print(p_yx.shape)
#         #p_yx = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
#         p_y = np.expand_dims(p_yx.mean(axis=0),0)
#         kl_d = p_yx*(np.log(p_yx+eps) - np.log(p_y+eps))
        
#         sum_kl_d = kl_d.sum(axis=1)
#         avg_kl_d = np.mean(sum_kl_d)
#         is_score = np.exp(avg_kl_d)
#         scores.append(is_score)
        
#     is_avg, is_std = np.mean(scores), np.std(scores)
#     return is_avg, is_std


# In[8]:


# np.random.shuffle(train_images)
# is_avg, is_std = calculate_IS(train_images)
# print('CIFAR10 score', is_avg, is_std)

# np.random.shuffle(dcgan_images)
# is_avg, is_std = calculate_IS(dcgan_images)
# print('DCGAN score', is_avg, is_std)

# np.random.shuffle(wgan_images2)
# is_avg, is_std = calculate_IS(wgan_images)
# print('WGAN score', is_avg, is_std)

