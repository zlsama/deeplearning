# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 09:39:25 2018

@author: zhanglisama    jxufe
"""

import matplotlib.pyplot as plt
import matplotlib.image as mping
import numpy as np
import tensorflow as tf

pic1=plt.imread('1.jpg')
plt.figure(1)
plt.imshow(pic1)
print(pic1.shape)


pic2=plt.imread('2.jpg')
plt.figure(2)
plt.imshow(pic2)
plt.show()
print(pic2.shape)

full1=np.reshape(pic1,[1,960,1080,3])
inputfull1=tf.Variable(tf.constant(1.0,shape=[1,960,1080,3]))
filter=tf.Variable(tf.constant([[-1.0,-1.0,-1.0],[0,0,0],[1.0,1.0,1.0],[-2.0,-2.0,-2.0],[0,0,0],[2.0,2.0,2.0],[-1.0,-1.0,-1.0],[0,0,0],[1.0,1.0,1.0]],shape=[3,3,3,1]))

op=tf.nn.conv2d(inputfull1,filter,strides=[1,1,1,1],padding='SAME')
o=tf.cast(  ((op-tf.reduce_min(op))/(tf.reduce_max(op)-tf.reduce_min(op)) ) *255 ,tf.uint8)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    t,f=sess.run([o,filter],feed_dict={inputfull1:full1})
    t=np.reshape(t,[960,1080])
    plt.imshow(t,cmap='Greys_r')
    plt.show()

