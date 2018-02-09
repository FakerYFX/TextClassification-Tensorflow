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
        self.rnn_size = config.rnn_size
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
        self.in_len = tf.placeholder(tf.int32, [None])
        self.in_x = tf.placeholder(tf.int32, [None, self.max_seq_len])  # shape: (batch x seq)
        self.in_y = tf.placeholder(tf.int32, [None])

        # Embedding layer
        # shape: (batch x seq x word_dim)
        embedded_seq = tf.nn.embedding_lookup(self.we, self.in_x)
        rnn_cell_fw = tf.nn.rnn_cell.GRUCell(self.rnn_size)
        rnn_cell_bw = tf.nn.rnn_cell.GRUCell(self.rnn_size)
        # outputs: A tuple (output_fw, output_bw)
        # output_fw: [batch_size, max_time, cell_bw.output_size]
        b_outputs, b_states = tf.nn.bidirectional_dynamic_rnn(rnn_cell_fw,
                                                              rnn_cell_bw,
                                                              embedded_seq, self.in_len,
                                                              dtype=tf.float32)
        # [batch_size, cell.state_size x 2]
        rnn_out = tf.concat(b_states, axis=-1)
        fc1 = tf.layers.dense(rnn_out, self.rnn_size, activation=tf.nn.relu)
        fc1_drop = tf.nn.dropout(fc1, keep_prob=self.drop_keep_rate)
        self.fc2 = tf.layers.dense(fc1_drop, self.mlp_size, activation=tf.nn.relu)
        self.logits = tf.layers.dense(self.fc2, self.class_num)

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
            self.in_len: batch[2],
            self.drop_keep_rate: drop_keep_rate
        }
        return_list = [self.train_op, self.merged, self.global_step, self.cost, self.acc_num]

        return sess.run(return_list, feed_dict)

    def model_test(self, sess, batch):
        feed_dict = {
            self.in_x: batch[0],
            self.in_y: batch[1],
            self.in_len: batch[2],
            self.drop_keep_rate: 1.0
        }
        return sess.run(self.acc_num, feed_dict)

    def model_get_represent(self, sess, batch):
        feed_dict = {
            self.in_x: batch[0],
            self.in_len: batch[2],
            self.drop_keep_rate: 1.0
        }
        return sess.run(self.fc2, feed_dict)

    def train(self, sess, train_xy, batch_size=32, summary_writer=None):
        windows = 1000
        cost_sum = 0.0
        acc_sum = 0
        sample_cnt = 0
        for batch in batch_iter(train_xy[0], train_xy[1], train_xy[2], batch_size=batch_size, shuffle=True):
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

    def test(self, sess, test_data, load_init_model=False):
        if load_init_model:
            ckpt = tf.train.get_checkpoint_state(self.model_dir + '/{}'.format(self.model_name))
            print('model file: ', ckpt.model_checkpoint_path)
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                self.saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                raise RuntimeError('not exist model ...')
        acc_sum = 0
        sample_sum = 0
        for batch in batch_iter(test_data[0], test_data[1], test_data[2], batch_size=100):
            acc_num = self.model_test(sess, batch)
            acc_sum += acc_num
            sample_sum += len(batch[1])
        return acc_sum/sample_sum

    def train_dev_test(self, sess, train_xy, dev_xy=None, test_xy=None,
                       batch_size=32, epoch_size=10, save_model=False, summary_writer=None):
        for epoch in range(epoch_size):
            cost, acc = self.train(sess, train_xy, batch_size, summary_writer)
            print ("Epoch {}: train cost: {}, acc: {}".format(epoch, cost, acc))
            if dev_xy is not None:
                dev_acc = self.test(sess, dev_xy)
                print("Epoch {}: dev acc: {}".format(epoch, dev_acc))
            if test_xy is not None:
                test_acc = self.test(sess, test_xy)
                print("Epoch {}: test acc: {}".format(epoch, test_acc))

            if save_model:
                save_file = self.model_dir+'/{}/{}_saver.ckpt'.format(self.model_name, self.model_name)
                self.saver.save(sess, save_file, global_step=epoch + 1)

            if epoch < 3:
                sess.run(self.learning_rate_decay_op)

    def get_represent(self, sess, train_xy):
        ckpt = tf.train.get_checkpoint_state(self.model_dir + '/{}'.format(self.model_name))
        print('model file: ', ckpt.model_checkpoint_path)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            self.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            raise RuntimeError('not exist model ...')

        represent = []
        for batch in batch_iter(train_xy[0], train_xy[1], train_xy[2], batch_size=100):
            repre = self.model_get_represent(sess, batch)
            represent += repre.tolist()
        return represent
