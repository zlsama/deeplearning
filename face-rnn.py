# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 19:17:15 2018

@author: zhanglisama    jxufe
"""

import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from PIL import Image
from time import time
#获取dataset
def load_data(dataset_path):
    
    X = []
    
    for dirname,dirnames,filensmes in os.walk(dataset_path):
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
        
    images_train,images_test,target_train,target_test=train_test_split(X,label,test_size=0.1,random_state=0)
    return [(images_train,images_test),(target_train,target_test)]
t0 = time()
dataset_path = 'att_faces'
dataset = load_data(dataset_path)
batch_size = 40
train_set_x = dataset[0][0]   
train_set_x =np.asarray(train_set_x, dtype='float32')
train_set_x = np.reshape(train_set_x,[-1,112,92])
train_set_y = dataset[1][0]    #训练标签
valid_set_x = dataset[0][1]
valid_set_x = np.asarray(valid_set_x, dtype='float32')
valid_set_x = np.reshape(valid_set_x,[-1,112,92])
valid_set_y = dataset[1][1]
n_input = 112
n_steps = 92
n_hidden = 120
n_classes = 40

X = tf.placeholder('float', [None, n_input, n_steps])  #x.shape --> (40,112,92)
Y = tf.placeholder('float', [None, n_classes])   

x1= tf.unstack(X, n_input, 1)  #  axis=1 按行进行拆分，共有112行，所以拆成112份，每一份为40x92
#lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
#outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x1,dtype=tf.float32)

##GRU
gru = tf.contrib.rnn.GRUCell(n_hidden)
outputs = tf.contrib.rnn.static_rnn(gru, x1, dtype=tf.float32)

predict = tf.contrib.layers.fully_connected(outputs[-1],n_classes,activation_fn=None)

learning_rate = 0.01

batch_size = 40

# Define loss and optimizer
cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost_func)

# Evaluate model
correct_pred = tf.equal(tf.argmax(predict,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
print('hello')
#启动session
with tf.Session() as session:
      session.run(tf.global_variables_initializer())
      for epoch in range(200):
            epoch_loss = 0
            for i in range((int)(np.shape(train_set_x)[0] / batch_size)):
                x = train_set_x[i * batch_size: (i + 1) * batch_size]
                y = train_set_y[i * batch_size: (i + 1) * batch_size]
                _, cost = session.run([optimizer, cost_func], feed_dict={X: x, Y: y})
                epoch_loss += cost
      correct = tf.equal(tf.argmax(predict,1), tf.argmax(Y,1))
      valid_accuracy = tf.reduce_mean(tf.cast(correct,'float'))
      print('valid set accuracy: ', valid_accuracy.eval({X: valid_set_x, Y: valid_set_y}))
      print('time:%f'%(time()-t0))