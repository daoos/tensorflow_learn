#first_session_only_tensorflow.py
# coding=utf8
import tensorflow as tf
# 定义一个常量x
x = tf.constant(1, name='x')
# 定义一个变量y=x+90
y = tf.Variable(x+90,name='y')

#初始化所有的变量，构建一个模型
model = tf.initialize_all_variables()
# 构建一个session
with tf.Session() as session:
    # 运行模型
    session.run(model)
    # 打印运行结果
    print(session.run(y))
