# coding=utf8
import tensorflow as tf

state = tf.Variable(0,name='counter')
# 打印依稀名字
print (state.name)
one = tf.constant(1)
new_value = tf.add(state,one)
# assgin更新变量，讲new_value的值赋值给state
update = tf.assign(state,new_value)
# 初始化所有的变量
init = tf.initialize_all_variables()
#
with tf.Session() as sess:
    # 加载初始值
    sess.run(init)
    # 循环3次
    for _ in range(30):
        # 获取更新的值
        sess.run(update)
        print sess.run(state)
