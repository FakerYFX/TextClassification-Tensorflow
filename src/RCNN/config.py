# coding : utf-8
import os
import numpy as np
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # for issue: The TensorFlow library wasn't compiled to use SSE3

random_seed = 1234
tf.set_random_seed(random_seed)
np.random.seed(random_seed)


class Config(object):
    # [General]
    _HOME_PATH = "/workspace/imagenet-data/DeepTextClassifier" 

    data_name = "Health"    # Yahoo!Answers-Health
    print _HOME_PATH
    train_file = _HOME_PATH + "/data/{}/train.txt".format(data_name)
    test_file = _HOME_PATH + "/data/{}/test.txt".format(data_name)
    dev_file = None
    model_dir = _HOME_PATH + "/data/model"
    word_embed_file = _HOME_PATH + "/data/embed/glove.6B.100d.txt"
    vocab2id_file_pkl = _HOME_PATH + "/data/embed/{}_vocab2id.pkl".format(data_name)
    we_file_pkl = _HOME_PATH + "/data/embed/{}_we.pkl".format(data_name)
    log_dir = _HOME_PATH + "/log"
    idf_file_pkl = _HOME_PATH + "/data/embed/{}_idf.pkl".format(data_name)
    # [model]
    we = None
    model_name = "cnn"
    epoch_size = 5
    batch_size = 128
    word_dim = 100
    max_seq_len = 16
    mlp_size = 50
    class_num = 21
    max_gradient_norm = 10
    learning_rate = 1e-3
    l2_rate = 1e-4


class ConfigAvg(object):
    # [General]
    _HOME_PATH = "/workspace/imagenet-data/DeepTextClassifier"

    data_name = "Health"    # Yahoo!Answer-Health

    train_file = _HOME_PATH + "/data/{}/train.txt".format(data_name)
    test_file = _HOME_PATH + "/data/{}/test.txt".format(data_name)
    dev_file = None
    model_dir = _HOME_PATH + "/data/model"
    word_embed_file = _HOME_PATH + "/data/embed/glove.6B.100d.txt"
    vocab2id_file_pkl = _HOME_PATH + "/data/embed/{}_vocab2id.pkl".format(data_name)
    we_file_pkl = _HOME_PATH + "/data/embed/{}_we.pkl".format(data_name)
    log_dir = _HOME_PATH + "/log"
    idf_file_pkl = _HOME_PATH + "/data/embed/{}_idf.pkl".format(data_name)
    # [model]
    we = None
    model_name = "TextAVG"
    epoch_size = 5
    batch_size = 128
    word_dim = 100
    max_seq_len = 16
    mlp_size = 50
    class_num = 21
    max_gradient_norm = 10
    learning_rate = 1e-3
    l2_rate = 1e-4


class ConfigAvgTFIDF(object):
    # [General]
    _HOME_PATH = "/workspace/imagenet-data/DeepTextClassifier"

    data_name = "Health"    # # Yahoo!Answer-Health

    train_file = _HOME_PATH + "/data/{}/train.txt".format(data_name)
    test_file = _HOME_PATH + "/data/{}/test.txt".format(data_name)
    dev_file = None
    model_dir = _HOME_PATH + "/data/model"
    word_embed_file = _HOME_PATH + "/data/embed/glove.6B.100d.txt"
    vocab2id_file_pkl = _HOME_PATH + "/data/embed/{}_vocab2id.pkl".format(data_name)
    we_file_pkl = _HOME_PATH + "/data/embed/{}_we.pkl".format(data_name)
    log_dir = _HOME_PATH + "/log"
    idf_file_pkl = _HOME_PATH + "/data/embed/{}_idf.pkl".format(data_name)
    # [model]
    we = None
    model_name = "TextAVG"
    epoch_size = 5
    batch_size = 128
    word_dim = 100
    max_seq_len = 16
    mlp_size = 50
    class_num = 21
    max_gradient_norm = 10
    learning_rate = 1e-3
    l2_rate = 1e-4


class ConfigRnn(object):
    # [General]
    _HOME_PATH = "/workspace/imagenet-data/DeepTextClassifier"

    data_name = "Health"    # # Yahoo!Answer-Health

    train_file = _HOME_PATH + "/data/{}/train.txt".format(data_name)
    test_file = _HOME_PATH + "/data/{}/test.txt".format(data_name)
    dev_file = None
    model_dir = _HOME_PATH + "/data/model"
    word_embed_file = _HOME_PATH + "/data/embed/glove.6B.100d.txt"
    vocab2id_file_pkl = _HOME_PATH + "/data/embed/{}_vocab2id.pkl".format(data_name)
    we_file_pkl = _HOME_PATH + "/data/embed/{}_we.pkl".format(data_name)
    log_dir = _HOME_PATH + "/log"
    idf_file_pkl = _HOME_PATH + "/data/embed/{}_idf.pkl".format(data_name)
    # [model]
    we = None
    model_name = "TextRNN"
    epoch_size = 5
    batch_size = 128
    word_dim = 100
    max_seq_len = 16
    rnn_size = 100
    mlp_size = 50
    class_num = 21
    max_gradient_norm = 10
    learning_rate = 1e-3
    l2_rate = 1e-4


class ConfigRnnCnn(object):
    # [General]
    _HOME_PATH = "/workspace/imagenet-data/DeepTextClassifier"

    data_name = "Health"    # # Yahoo!Answer-Health

    train_file = _HOME_PATH + "/data/{}/train.txt".format(data_name)
    test_file = _HOME_PATH + "/data/{}/test.txt".format(data_name)
    dev_file = None
    model_dir = _HOME_PATH + "/data/model"
    word_embed_file = _HOME_PATH + "/data/embed/glove.6B.100d.txt"
    vocab2id_file_pkl = _HOME_PATH + "/data/embed/{}_vocab2id.pkl".format(data_name)
    we_file_pkl = _HOME_PATH + "/data/embed/{}_we.pkl".format(data_name)
    log_dir = _HOME_PATH + "/log"
    idf_file_pkl = _HOME_PATH + "/data/embed/{}_idf.pkl".format(data_name)
    # [model]
    we = None
    model_name = "rnn_cnn"
    epoch_size = 5
    batch_size = 128
    word_dim = 100
    max_seq_len = 16
    rnn_size = 100
    mlp_size = 50
    class_num = 21
    max_gradient_norm = 10
    learning_rate = 1e-3
    l2_rate = 1e-4


class ConfigRcnn(object):
    # [General]
    _HOME_PATH = "/workspace/imagenet-data/DeepTextClassifier"

    data_name = "Health"    # # Yahoo!Answer-Health

    train_file = _HOME_PATH + "/data/{}/train.txt".format(data_name)
    test_file = _HOME_PATH + "/data/{}/test.txt".format(data_name)
    dev_file = None
    model_dir = _HOME_PATH + "/data/model"
    word_embed_file = _HOME_PATH + "/data/embed/glove.6B.100d.txt"
    vocab2id_file_pkl = _HOME_PATH + "/data/embed/{}_vocab2id.pkl".format(data_name)
    we_file_pkl = _HOME_PATH + "/data/embed/{}_we.pkl".format(data_name)
    log_dir = _HOME_PATH + "/log"
    idf_file_pkl = _HOME_PATH + "/data/embed/{}_idf.pkl".format(data_name)
    # [model]
    we = None
    model_name = "rcnn"
    epoch_size = 5
    batch_size = 128
    word_dim = 100
    max_seq_len = 16
    rnn_size = 128
    mlp_size = 50
    class_num = 21
    max_gradient_norm = 10
    learning_rate = 1e-3
    l2_rate = 1e-4


class ConfigRnnAtt(object):
    # [General]
    _HOME_PATH = "/workspace/imagenet-data/DeepTextClassifier"

    data_name = "Health"    # # Yahoo!Answer-Health

    train_file = _HOME_PATH + "/data/{}/train.txt".format(data_name)
    test_file = _HOME_PATH + "/data/{}/test.txt".format(data_name)
    dev_file = None
    model_dir = _HOME_PATH + "/data/model"
    word_embed_file = _HOME_PATH + "/data/embed/glove.6B.100d.txt"
    vocab2id_file_pkl = _HOME_PATH + "/data/embed/{}_vocab2id.pkl".format(data_name)
    we_file_pkl = _HOME_PATH + "/data/embed/{}_we.pkl".format(data_name)
    log_dir = _HOME_PATH + "/log"
    idf_file_pkl = _HOME_PATH + "/data/embed/{}_idf.pkl".format(data_name)
    # [model]
    we = None
    model_name = "rnn-att"
    epoch_size = 5
    batch_size = 128
    word_dim = 100
    max_seq_len = 16
    rnn_size = 128
    mlp_size = 50
    class_num = 21
    max_gradient_norm = 10
    learning_rate = 1e-3
    l2_rate = 1e-4
