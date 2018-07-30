# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 20:35:04 2018

@author: zhanglisama    jxufe
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)
tf.reset_default_graph
x=tf.placeholder(tf.float32,[None,784])
w1=tf.Variable(tf.truncated_normal([784,300],stddev=0.1))
b1=tf.Variable(tf.zeros([300]))
w2=tf.Variable(tf.zeros([300,10]))
b2=tf.Variable(tf.zeros([10]))
y=tf.placeholder(tf.float32,[None,10])
#y_pred=tf.nn.softmax(tf.nn.tanh(tf.matmul(x,w)+b))
y_=tf.nn.relu(tf.matmul(x,w1)+b1)
y_pred=tf.nn.softmax(tf.matmul(y_,w2)+b2)
keep_prob=tf.placeholder(tf.float32)
#loss=tf.reduce_mean(-tf.reduce_sum(y*tf.log(y_pred)+(1-y)*tf.log(1-y_pred),reduction_indices=[1]))

loss=tf.reduce_mean(-tf.reduce_sum(y*tf.log(y_pred),reduction_indices=[1]))

optimizer=tf.train.AdagradOptimizer(0.3).minimize(loss)
init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(3000):
        batch_xs,batch_ys=mnist.train.next_batch(100)
        sess.run(optimizer,feed_dict={x:batch_xs,y:batch_ys,keep_prob:0.75})
    correct=tf.equal(tf.argmax(y,1),tf.argmax(y_pred,1))
    accuracy=tf.reduce_mean(tf.cast(correct,tf.float32))
    print('train:',accuracy.eval({x:mnist.train.images,y:mnist.train.labels}))
    print('test:',accuracy.eval({x:mnist.test.images,y:mnist.test.labels,keep_prob:1}))
    
        
    