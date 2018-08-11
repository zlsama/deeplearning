# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 09:18:07 2018

@author: zhanglisama    jxufe
"""

from PIL import Image
import numpy as np
import os
from time import time
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf

sess=tf.InteractiveSession()
def read_images(path):
    X = []
    
    for dirname,dirnames,filensmes in os.walk(path):
        for subdirname in dirnames:   
            subject_path=os.path.join(dirname,subdirname)  
            for filename in os.listdir(subject_path):   
                im=Image.open(os.path.join(subject_path,filename))
                im=np.asarray(im,dtype=np.float32)/256
                im=np.ndarray.flatten(im)
                X.append(im)
                
    label = np.zeros((400, 40))
    for i in range(40):
        label[i * 10: (i + 1) * 10, i] = 1
    return [X,label]

data=read_images('att_faces')        
ima_data=data[0]
ima_data = np.asarray(ima_data, dtype='float32')



target=data[1]

images_train,images_test,target_train,target_test=train_test_split(ima_data,target,test_size=0.2,random_state=0)


def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x,w):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')

def pool(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

batch_size=40

x=tf.placeholder(tf.float32,[None,112*92])
y=tf.placeholder(tf.float32,[None,40])

x1=tf.reshape(x,[-1,112,92,1])
w_conv1=weight_variable([5,5,1,32])
b_conv1=bias_variable([32])



h_conv1=tf.nn.relu(conv2d(x1,w_conv1)+b_conv1)
h_pool1=pool(h_conv1)

w_conv2=weight_variable([5,5,32,64])
b_conv2=bias_variable([64])

h_conv2=tf.nn.relu(conv2d(h_pool1,w_conv2)+b_conv2)
h_pool2=pool(h_conv2)

w_fc1=weight_variable([28*23*64,1024])                     
b_fc1=bias_variable([1024])
h_pool = tf.contrib.layers.flatten(h_pool2)
h_fc1=tf.nn.relu(tf.matmul(h_pool ,w_fc1)+b_fc1)

keep_prob=tf.placeholder(tf.float32)
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)

w_fc2=weight_variable([1024,40])
b_fc2=bias_variable([40])
y_conv=tf.matmul(h_fc1,w_fc2)+b_fc2

cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y))
train_step=tf.train.AdamOptimizer(0.001).minimize(cross_entropy) 

tf.global_variables_initializer().run()


for epoch in range(20):
    for i in range((int)(np.shape(images_train)[0] / batch_size)):
        x_ = images_train[i * batch_size: (i + 1) * batch_size]
        y_ = target_train[i * batch_size: (i + 1) * batch_size]
        train_step.run( feed_dict={x: x_, y: y_,})
    correct=tf.equal(tf.argmax(y,1),tf.argmax(y_conv,1))
    accuracy=tf.reduce_mean(tf.cast(correct,tf.float32))
print('test accuracy: ', accuracy.eval({x: images_test, y: target_test}))