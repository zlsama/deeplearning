# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 14:42:35 2018

@author: zhanglisama    jxufe
"""

import os
import numpy as np 
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split

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
        
    images_train,images_test,target_train,target_test=train_test_split(X,label,test_size=0.2,random_state=0)
    return [(images_train,images_test),(target_train,target_test)]
	#返回训练数据集、测试数据集、训练集标签、测试集标签，images_train和images_test类型
	#为list，需要利用np.asarray()转换类型。

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

def train_facedata(dataset):
    batch_size = 40
    train_set_x = dataset[0][0]   
    train_set_x =np.asarray(train_set_x, dtype='float32')
    train_set_y = dataset[1][0]    #训练标签
    valid_set_x = dataset[0][1]
    valid_set_x =np.asarray(valid_set_x, dtype='float32')
    valid_set_y = dataset[1][1]
    X = tf.placeholder(tf.float32, [None, 112*92])
    Y = tf.placeholder(tf.float32, [None, 40])
    w_conv1=weight_variable([5,5,1,32])
    b_conv1=bias_variable([32])
    X1 = tf.reshape(X,[-1,112,92,1])
    
    h_conv1=tf.nn.relu((conv2d(X1,w_conv1)+b_conv1))
    h_pool1=pool(h_conv1)
    w_conv2 = weight_variable([5,5,32,64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1,w_conv2)+b_conv2)
    h_pool2 = pool(h_conv2)
    w_conv3 = weight_variable([3,3,64,32])
    b_conv3 = bias_variable([32])
    h_conv  = tf.nn.relu(conv2d(h_pool2,w_conv3)+b_conv3)
    h_pool = pool(h_conv)
    h_pool = tf.contrib.layers.flatten(h_pool)
    w_fc1 = weight_variable([14*12*32,1024])
    b_fc1 = bias_variable([1024])
    w_fc2 = weight_variable([1024,40])
    b_fc2 = bias_variable([40])
    h_fc1 = tf.add(tf.matmul(h_pool,w_fc1),b_fc1)
    predict = tf.matmul(h_fc1,w_fc2)+b_fc2
    cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=Y))
    optimizer = tf.train.AdamOptimizer(1e-3).minimize(cost_func)
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        for epoch in range(20):
            epoch_loss = 0
            for i in range((int)(np.shape(train_set_x)[0] / batch_size)):
                x = train_set_x[i * batch_size: (i + 1) * batch_size]
                y = train_set_y[i * batch_size: (i + 1) * batch_size]
                _, cost = session.run([optimizer, cost_func], feed_dict={X: x, Y: y})
                epoch_loss += cost

        correct = tf.equal(tf.argmax(predict,1), tf.argmax(Y,1))
        valid_accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        print('valid set accuracy: ', valid_accuracy.eval({X: valid_set_x, Y: valid_set_y}))

def main():
    dataset_path = 'att_faces'
    data = load_data(dataset_path)
    train_facedata(data)
   

if __name__ == "__main__" :
    main()