# coding=utf8
import tensorflow as tf

matrix1 = tf.constant([[3,3]])
matrix2 = tf.constant([[2],[2]])
# 矩阵乘法运算
prouct = tf.matmul(matrix1,matrix2)


# sess = tf.Session()
# reslut = sess.run(prouct)
# print(reslut)
# sess.close()

with tf.Session() as sess:
    result2 = sess.run(prouct)
    print(result2)