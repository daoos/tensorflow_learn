# coding=utf8
import tensorflow as tf
import numpy as np

x_data=np.random.rand(100).astype(np.float32)
y_data=x_data*0.1+0.3
# 生成随机数列
Weight = tf.Variable(tf.random_uniform([1],-1.0,1.0))
biases = tf.Variable(tf.zeros([1]))

# 定义Y值
y = Weight*x_data+biases
#
loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)
# 初始化参数
init = tf.initialize_all_variables()
# session就像一个指针，用于激活向量运算
sess = tf.Session()
sess.run(init)

for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print (step, sess.run(Weight),sess.run(biases))


