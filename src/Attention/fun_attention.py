# coding: utf-8
from __future__ import print_function
import tensorflow as tf


def attention_self(in_x, in_len, in_dim, u_dim, name=0):
    in_x_shape = tf.shape(in_x)  # [batch_size, max_time, h_size]
    in_x_temp = tf.reshape(in_x, [in_x_shape[0]*in_x_shape[1], in_x_shape[2]])
    w_x = tf.get_variable('w_self_hidden_%d' % name,
                          [in_dim, u_dim], initializer=tf.random_normal_initializer(stddev=0.01))
    b_x = tf.get_variable('b_self_hidden_%d' % name, [u_dim], initializer=tf.zeros_initializer)
    u = tf.nn.xw_plus_b(in_x_temp, w_x, b_x)  # [(batch*max_time) x u_dim]
    u = tf.tanh(u)  # [(batch x time_step) x u_dim]

    w_u = tf.get_variable('w_self_u_%d' % name, [u_dim, 1], initializer=tf.random_normal_initializer(stddev=0.01))
    att = tf.matmul(u, w_u)  # [(batch x time_step)x1]
    att = tf.reshape(att, [in_x_shape[0], in_x_shape[1]])   # [batch x max_time]
    att = tf.exp(att) * tf.sequence_mask(in_len, maxlen=in_x_shape[1], dtype=tf.float32)  # [batch x max_time]
    att_sum = tf.reduce_sum(att, axis=1, keep_dims=True)  # [batch x 1]
    att_prob = att / att_sum  # [batch x time_step]
    return att_prob

