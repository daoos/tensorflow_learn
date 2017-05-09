# coding=utf8
import tensorflow as tf
# 1. 保存计算两个变量和的模型。
v1 = tf.Variable(tf.constant(1.0, shape=[1]), name = "v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name = "v2")
result = v1 + v2

init_op = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init_op)
    saver.save(sess, "Saved_model/model.ckpt")
# 2. 加载保存了两个变量和的模型。
with tf.Session() as sess:
    saver.restore(sess, "Saved_model/model.ckpt")
    print sess.run(result)
# 3. 直接加载持久化的图。
saver = tf.train.import_meta_graph("Saved_model/model.ckpt.meta")
with tf.Session() as sess:
    saver.restore(sess, "Saved_model/model.ckpt")
    print sess.run(tf.get_default_graph().get_tensor_by_name("add:0"))
# 4. 变量重命名。
v1 = tf.Variable(tf.constant(1.0, shape=[1]), name = "other-v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name = "other-v2")
saver = tf.train.Saver({"v1": v1, "v2": v2})