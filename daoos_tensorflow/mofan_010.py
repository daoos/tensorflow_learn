# coding=utf8
# 约定矩阵大写字母开头，列表其他变量使用小写
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# 定义一个添加神经层的函数
def add_layer(inputs, in_size, out_size, activation_function=None):
    # 生成一个随机变量，比初始化都是0或者1要效果好很多
    Weights = tf.Variable(tf.random_uniform([in_size, out_size]))
    # 添加一个偏移量，一般推荐不使用0所以+0.1
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    # 定义输入值乘以随机变量再加上偏移量作为运算后的结果,预测的结果值
    Wx_puls_b = tf.matmul(inputs, Weights) + biases
    # 如果激励函数为None则直接返回结果
    if activation_function is None:
        outputs = Wx_puls_b
    else:
        # 否则使用激励函数进行运算
        outputs = activation_function(Wx_puls_b)
    # 最总返回结果输出值
    return outputs


# 有300个例子，
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
# 构建一个噪点，模拟真实数据，使用和x_data一样的数据格式，从0开始的方差为0.05
noise = np.random.normal(0, 0.05, x_data.shape)
# x_data的二次方
y_data = np.square(x_data) - 0.5 + noise
#
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])
# 添加一个隐藏神经层(一个隐藏层10个神经元，激励函数为tf.nn.relu)
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
# 定义一个输出层
predition = add_layer(l1, 10, 1, activation_function=None)
# 对于没一个例子差再求和再对求和后的数值再进行求平均值
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - predition), reduction_indices=[1]))
# 通过优化器，减小误差
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
# 初始化所有参数
init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    # 定义一个框
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x_data, y_data)
    # 如果连续执行不暂停使用ion()
    plt.ion()
    #
    plt.show()
    # 循环1000步
    for i in range(1000):
        #
        sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
        # 每五十步打印一次误差值
        if i % 50 == 0:
            # 打印误差值
            print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
            try:
                # 先清除旧的线
                ax.lines.remove(lines[0])
            except Exception:
                pass
            # 获取运行中的值
            prediction_value = sess.run(predition, feed_dict={xs: x_data})
            # 设置线的数值和属性，红色，宽度为5，x轴为x_data,y轴为prediction_value
            lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
            # 暂停0.1秒
            plt.pause(0.1)

