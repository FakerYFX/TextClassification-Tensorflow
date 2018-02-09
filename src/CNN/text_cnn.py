# coding: utf-8
from __future__ import print_function
from __future__ import division
import tensorflow as tf
from tools import batch_iter


class Model(object):
    def __init__(self, config):
        # base config
        self.model_name = config.model_name
        self.model_dir = config.model_dir
        self.log_dir = config.log_dir

        # Setup Model Parameters
        self.max_seq_len = config.max_seq_len
        # self.vocab_size = config.vocab_size
        self.word_dim = config.word_dim
        self.mlp_size = config.mlp_size
        self.class_num = config.class_num
        self.max_gradient_norm = config.max_gradient_norm
        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        self.learning_rate = tf.Variable(config.learning_rate, name='learning_rate', trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate.value()*0.1)

        self.drop_keep_rate = tf.placeholder(tf.float32, name='drop_keep_rate')

        # Word Embedding
        self.we = tf.Variable(config.we, name='emb')

        # Build the Computation Graph
        self._build_model()
        # Set cost
        self._set_cost_and_optimize()
        # Set prediction and acc
        self._set_predict()
        # add tensor board
        self._log_summaries()
        # model parameter saver
        self.saver = tf.train.Saver(tf.global_variables())

    def _build_model(self):
        # Model PlaceHolder for input
        self.in_x = tf.placeholder(tf.int32, [None, self.max_seq_len])  # shape: (batch x seq)
        self.in_y = tf.placeholder(tf.int32, [None])

        # Embedding layer
        # shape: (batch x seq x word_dim)
        embedded_seq = tf.nn.embedding_lookup(self.we, self.in_x)

        # shape: (batch x seq x word_dim x 1)
        embedded_seq = tf.expand_dims(embedded_seq, -1)
        print("embedded_seq:"+str(embedded_seq.shape))
        # Create a convolution + max_pool layer for each filter size
        pooled_outputs = []
        kernel_sizes = [1, 2, 3, 4, 5]
        kernel_nums = [64, 64, 64, 64, 64]
        num_filters_total = 64*5
        for i in range(len(kernel_sizes)):
            with tf.name_scope("convolution-max-pool-%s" % kernel_sizes[i]):
                # convolution Layer
                filter_shape = [kernel_sizes[i], self.word_dim, 1, kernel_nums[i]]
                w_filter = tf.Variable(tf.truncated_normal(shape=filter_shape, stddev=0.1))
                b = tf.Variable(tf.constant(0.0, shape=[kernel_nums[i]]))
                conv = tf.nn.conv2d(embedded_seq, w_filter, strides=[1, 1, 1, 1], padding="VALID", name="conv")
                # h shape: (batch x seq-filter_size[i]+1 x 1 x filter_num[i])
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # pooled shape: (batch x 1 x 1, filter_num[i])
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.max_seq_len - kernel_sizes[i] + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        cnn_pool = tf.concat(pooled_outputs, 3)   # (batch x 1 x 1 x num_filters_total)
        print("cnn_pool:"+str(cnn_pool))
	cnn_pool_flat = tf.reshape(cnn_pool, [-1, num_filters_total])   # (batch x num_filters_total)
        cnn_pool_drop = tf.nn.dropout(cnn_pool_flat, self.drop_keep_rate)

        # hidden layer
        w_h = tf.Variable(tf.random_normal([num_filters_total, self.mlp_size], stddev=0.01))
        b_h = tf.Variable(tf.zeros([self.mlp_size]))
        layer_fc = tf.nn.tanh(tf.nn.xw_plus_b(cnn_pool_drop, w_h, b_h))

        # logits layer
        w = tf.Variable(tf.random_normal([self.mlp_size, self.class_num], stddev=0.01))
        b = tf.Variable(tf.zeros([self.class_num]))
        self.logits = tf.nn.xw_plus_b(layer_fc, w, b)

    def _set_cost_and_optimize(self):
        softmax_cost = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.in_y))
        self.cost = softmax_cost
        optimizer = tf.train.AdamOptimizer(self.learning_rate)  # .minimize(self.cost)

        train_vars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, train_vars), self.max_gradient_norm)
        self.train_op = optimizer.apply_gradients(list(zip(grads, train_vars)),
                                                  global_step=self.global_step)

    def _set_predict(self):
        y_prob = tf.nn.softmax(self.logits)
        self.y_p = tf.cast(tf.argmax(y_prob, 1), tf.int32)
        # Accuracy
        check_prediction = tf.equal(self.y_p, self.in_y)
        self.acc_num = tf.reduce_sum(tf.cast(check_prediction, tf.int32))
        self.acc = tf.reduce_mean(tf.cast(check_prediction, tf.float32))

    def _log_summaries(self):
        """
        Adds summaries for the following variables to the graph and returns
        an operation to evaluate them.
        """
        cost = tf.summary.scalar("cost", self.cost)
        acc = tf.summary.scalar("acc", self.acc)
        self.merged = tf.summary.merge([cost, acc])

    def model_train(self, sess, batch, drop_keep_rate=0.5):
        feed_dict = {
            self.in_x: batch[0],
            self.in_y: batch[1],
            self.drop_keep_rate: drop_keep_rate
        }
        return_list = [self.train_op, self.merged, self.global_step, self.cost, self.acc_num]

        return sess.run(return_list, feed_dict)

    def model_test(self, sess, batch):
        feed_dict = {
            self.in_x: batch[0],
            self.in_y: batch[1],
            self.drop_keep_rate: 1.0
        }
        return sess.run(self.acc_num, feed_dict)

    def train(self, sess, train_xy, batch_size=32, summary_writer=None):
        windows = 1000
        cost_sum = 0.0
        acc_sum = 0
        sample_cnt = 0
        for batch in batch_iter(train_xy[0], train_xy[1], batch_size=batch_size, shuffle=True):
            _, summary, step, cost, acc_num = self.model_train(sess, batch)
            summary_writer.add_summary(summary, step)
            sample_cnt += len(batch[1])
            if sample_cnt % windows:
                cost_sum += cost
                acc_sum += acc_num
            else:
                cost_sum = 0.0
                acc_sum = 0
                sample_cnt = 0
            # print ("cost: {}".format(cost))
        return cost_sum/sample_cnt, acc_sum/sample_cnt

    def test(self, sess, test_xy, load_init_model=False):
        if load_init_model:
            ckpt = tf.train.get_checkpoint_state(self.model_dir + '/{}'.format(self.model_name))
            print('model file: ', ckpt.model_checkpoint_path)
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                self.saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                raise RuntimeError('not exist model ...')
        acc_sum = 0
        sample_sum = 0
        for batch in batch_iter(test_xy[0], test_xy[1], batch_size=100):
            acc_num = self.model_test(sess, batch)
            acc_sum += acc_num
            sample_sum += len(batch[1])
        return acc_sum/sample_sum

    def train_dev_test(self, sess, train_xy, dev_xy=None, test_xy=None,
                       batch_size=32, epoch_size=10, save_model=False, summary_writer=None):
        for epoch in range(epoch_size):
            cost, acc = self.train(sess, train_xy, batch_size, summary_writer)
            print ("Epoch {}: cost: {}, acc: {}".format(epoch, cost, acc))
            if dev_xy is not None:
                dev_acc = self.test(sess, dev_xy)
                print("Epoch {}: dev acc: {}".format(epoch, dev_acc))
            if test_xy is not None:
                test_acc = self.test(sess, test_xy)
                print("Epoch {}: dev acc: {}".format(epoch, test_acc))

            if save_model:
                save_file = self.model_dir+'/{}/{}_saver.ckpt'.format(self.model_name, self.model_name)
                self.saver.save(sess, save_file, global_step=epoch + 1)

            if epoch < 3:
                sess.run(self.learning_rate_decay_op)
