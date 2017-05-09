# coding=utf8
# 1. 三层简单神经网络
import tensorflow as tf
## 1.1 定义变量
# 声明w1/w2两个变量，还通过seed设置了随机种子，
# random_normal函数正态分布的数据
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))
# 暂时将输入的特征向量定义为常量，x是一个1*2的矩阵
x = tf.constant([[0.7, 0.9]])
## 1.2 定义前向传播的神经网络
# 通过向前传播算法获得神经网络的输出
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)
# 1.3 调用会话输出结果
sess = tf.Session()
# 不能直接运行sess.run(y)来获取y的值
# 因为w1和w2都没有初始化，分别初始化
sess.run(w1.initializer)
sess.run(w2.initializer)
# 运行输出y的结果
print(sess.run(y))
sess.close()
# 2. 使用placeholder
x = tf.placeholder(tf.float32, shape=(1, 2), name="input")
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)
config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
sess = tf.Session(config=config)

init_op = tf.global_variables_initializer()
sess.run(init_op)

print(sess.run(y, feed_dict={x: [[0.7,0.9]]}))

# 3. 增加多个输入
x = tf.placeholder(tf.float32, shape=(3, 2), name="input")
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

sess = tf.Session()
#使用tf.global_variables_initializer()来初始化所有的变量
init_op = tf.global_variables_initializer()
sess.run(init_op)

print(sess.run(y, feed_dict={x: [[0.7,0.9],[0.1,0.4],[0.5,0.8]]}))