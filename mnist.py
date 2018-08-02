# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 14:42:35 2018

@author: zhanglisama    jxufe
"""

from tensorflow.examples.tutorials.mnist import  input_data
from time import time
import tensorflow as tf
mnist=input_data.read_data_sets('MNIST_data/',one_hot=True)

x=tf.placeholder(tf.float32,[None,784])
y=tf.placeholder(tf.float32,[None,10])
w1=tf.Variable(tf.truncated_normal([784,256],stddev=0.1))
b1=tf.Variable(tf.zeros([256]))
layer1=tf.nn.relu(tf.matmul(x,w1)+b1)

w2=tf.Variable(tf.truncated_normal([256,256],stddev=0.1))
b2=tf.Variable(tf.zeros([256]))
layer2=tf.nn.relu(tf.matmul(layer1,w2)+b2)

w_outlayer=tf.Variable(tf.truncated_normal([256,10],stddev=0.1))
b_outlayer=tf.Variable(tf.zeros([10]))
y_pred=tf.nn.softmax(tf.matmul(layer2,w_outlayer)+b_outlayer)

cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred,labels=y))
keep_prob=tf.placeholder(tf.float32)
optimizer=tf.train.AdamOptimizer(0.001).minimize(cost)

t0=time()
init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        batch_xs,batch_ys=mnist.train.next_batch(100)
        sess.run(optimizer,feed_dict={x:batch_xs,y:batch_ys,keep_prob:0.5})
    correct=tf.equal(tf.argmax(y,1),tf.argmax(y_pred,1))
    accuracy=tf.reduce_mean(tf.cast(correct,tf.float32))
    print('train:',accuracy.eval({x:mnist.train.images,y:mnist.train.labels}))
    print('test:',accuracy.eval({x:mnist.test.images,y:mnist.test.labels,keep_prob:1}))
    print('time: %f'%(time()-t0))