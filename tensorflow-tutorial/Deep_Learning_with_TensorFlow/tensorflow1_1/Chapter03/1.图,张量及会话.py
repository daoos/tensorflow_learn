# coding=utf8
# 1. 定义两个不同的图
# 不同的图(graph)中的数据是相互隔离的，以下举例两个不同的图中的相同变量名的数值是单独存储的
import tensorflow as tf
# 构建图
g1 = tf.Graph()
# 使用默认的图
with g1.as_default():
    # 构建一个初始值为0的变量
    v = tf.get_variable("v", [1], initializer = tf.zeros_initializer()) # 设置初始值为0
# 构建第二个图对象
g2 = tf.Graph()
# 使用默认图对象
with g2.as_default():
    # 构建第二个变量初始值为1
    v = tf.get_variable("v", [1], initializer = tf.ones_initializer())  # 设置初始值为1
#使用 with方式会自动关闭session,不用再调用session.close()
with tf.Session(graph = g1) as sess:
    # 使用session构建图g1,并舒适化参数
    tf.global_variables_initializer().run()
    #
    with tf.variable_scope("", reuse=True):
        # 打印图g1中的v的值
        print(sess.run(tf.get_variable("v")))
# 构建第二个图
with tf.Session(graph = g2) as sess:
    # 初始化第二个图中的所有变量
    tf.global_variables_initializer().run()
    #
    with tf.variable_scope("", reuse=True):
        # 打印第二个图中的变量v的值
        print(sess.run(tf.get_variable("v")))
# 2.张量的概念
a = tf.constant([1.0, 2.0], name="a")
b = tf.constant([2.0, 1.0], name="b")
result = a + b
print result

sess = tf.InteractiveSession ()
print(result.eval())
sess.close()
# 3. 会话的使用
##3.1 创建和关闭会话
# 创建一个会话。
sess = tf.Session()
a = tf.constant([1.0, 2.0], name="a")
b = tf.constant([2.0, 1.0], name="b")
result = a + b
# 使用会话得到之前计算的结果。
print(sess.run(result))

# 关闭会话使得本次运行中使用到的资源可以被释放。
sess.close()
##3.2 使用with statement 来创建会话
with tf.Session() as sess:
    print(sess.run(result))
# 3.3 指定默认会话
sess = tf.Session()
with sess.as_default():
     print(result.eval())
sess = tf.Session()

# 下面的两个命令有相同的功能。
print(sess.run(result))
print(result.eval(session=sess))
# 4. 使用tf.InteractiveSession构建会话
# 使用InteractiveSession实现交互，会自动将生成的会话注册为默认会话
sess = tf.InteractiveSession ()
print(result.eval())
sess.close()
# 5. 通过ConfigProto配置会话
# 通过配置去设置session会话，allow_soft_placement为True时是否允许在缺失GPU的时候使用CPU去运算，
# log_device_placement参数为True时开启打印每个节点安排到了哪个设备上
config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
# 使用可交互session
sess1 = tf.InteractiveSession(config=config)
sess2 = tf.Session(config=config)