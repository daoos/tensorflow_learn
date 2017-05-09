#first_session.py
# coding=utf8
#a simple Python code
#一个python程序
x = 1
y = x + 9
print(y)

#....and the tensorflow translation of the previous code
# 引入tensorflow方便操作
import tensorflow as tf
# 定义一个常量x
x = tf.constant(1, name='x')
# 定义一个变量y=x+9
y = tf.Variable(x+9,name='y')
print(y)
