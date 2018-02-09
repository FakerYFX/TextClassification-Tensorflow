# coding: utf-8
from __future__ import print_function
from __future__ import division
import tensorflow as tf

from text_self_structure_att import Model
from dataset import DataSet
from config import ConfigRnnAtt as Config
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

def main():
    my_config = Config()
    my_data = DataSet(my_config, True)
    my_config.we = my_data.we
    my_model = Model(my_config)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    summary_writer = tf.summary.FileWriter(my_config.log_dir, sess.graph)
    my_model.train_dev_test(sess, [my_data.train_x, my_data.train_y, my_data.train_seq_len],
                            [my_data.test_x, my_data.test_y, my_data.test_seq_len],
                            summary_writer=summary_writer)


if __name__ == '__main__':
    main()
