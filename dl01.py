# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 10:29:50 2018

@author: zlsama
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

train_x=np.linspace(-1,1,100)
train_y=2*train_x+np.random.randn(*train_x.shape)*0.3

plt.plot(train_x,train_y,'ro',label='Original data')
plt.legend()
plt.show()

X=tf.placeholder('float')
Y=tf.placeholder('float')
w=tf.Variable(tf.random_normal([1]),name='weight')
b=tf.Variable(tf.zeros([1]),name='bias')
z=tf.multiply(X,w)+b

cost=tf.reduce_mean(tf.square(Y-z))
learning_rate=0.01
optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init=tf.global_variables_initializer()
training_epochs=20
display_step=2

plotdata={'batchsize':[],'loss':[]}
def moving_average(a,w=10):
    if len(a)<w:
        return a[:]
    return [val if idx<w else sum(a[(idx-w):idx])/w for idx,val in enumerate(a)]

with tf.Session() as sess:
    sess.run(init)
    plotdata={'batchsize':[],'loss':[]}
    for epoch in range(training_epochs):
        for (x,y) in zip(train_x,train_y):
            sess.run(optimizer,feed_dict={X:x,Y:y})
            
            
        if epoch%display_step==0:
            loss=sess.run(cost,feed_dict={X:train_x,Y:train_y})
            print('epoch:',epoch+1,'cost=',loss,'w=',sess.run(w),'b=',sess.run(b))
            if not (loss=='NA'):
                plotdata['batchsize'].append(epoch)
                plotdata['loss'].append(loss)
    print('finished')
    print('cost=',sess.run(cost,feed_dict={X:train_x,Y:train_y}),'w=',sess.run(w),'b=',sess.run(b))
                
    plt.plot(train_x,train_y,'ro',label='Original data')
    plt.plot(train_x,sess.run(w)*train_x+sess.run(b),label='fittedline')
    plt.legend()
    plt.show()
    
    plotdata['avgloss']=moving_average(plotdata['loss'])
    plt.figure(1)
    plt.subplot(211)
    plt.plot(plotdata['batchsize'],plotdata['avgloss'],'b--')
    plt.ylabel('loss')
    plt.xlabel('minibatch number')
    plt.title('Minibatch run vs. training loss')
    plt.show()

      