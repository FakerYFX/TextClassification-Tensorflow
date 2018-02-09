# coding: utf-8
from __future__ import print_function
import numpy as np
import tools

#g_category2id = {'LOC': 0, 'HUM': 1, 'NUM': 2, 'ABBR': 3, 'ENTY': 4, 'DESC': 5}
g_divide = "\t"


def clean_str(in_str):
    return in_str.lower().strip().split()


def idf(in_files):
    all_num = 1
    idf_value = {}

    for in_file in in_files:
        with open(in_file, 'r') as fr:
            for line in fr:
                all_num += 1
                s_str = line.split(g_divide)[1]
                s = set(s_str.split())
                for w in s:
                    w = w.lower()
                    if w in idf_value:
                        idf_value[w] += 1
                    else:
                        idf_value[w] = 1

    for key in idf_value:
        n_w = idf_value[key]
        idf_value[key] = np.log(all_num / (1.0 + n_w))
        print ("{} : {}".format(key, idf_value[key]))
    print('==> finished calculate IDF.')
    return idf_value


def load_embed_from_text(in_file, token_dim):
    """
    :return: embed numpy, vocab2id dict
    """
    embed = []
    vocab2id = {}
    print('==> loading embed from txt')
    word_id = 0
    embed.append([0.0] * token_dim)
    word_id += 1
    with open(in_file, 'r') as fr:
        # print('embedding info: ', fr.readline())
        for line in fr:
            t = line.split()
            embed.append(map(float, t[1:]))
            vocab2id[t[0]] = word_id
            word_id += 1
    print('==> finished load input embed from txt')
    return np.array(embed, dtype=np.float32), vocab2id


def sentence2id_and_pad(in_file, vocab2id, max_sent_len):
    x = []
    y = []
    seq_len = []
    miss_tokens = 0
    with open(in_file, 'r') as fr:
        for line in fr:
            t = line.strip().split(g_divide)
            y.append(t[0])
            words = clean_str(t[1])
            words = words[: max_sent_len]
            seq_len.append(len(words))
            t = [0] * max_sent_len
            for i, w in enumerate(words):
                if i == max_sent_len:
                    break
                t[i] = vocab2id.get(w, 0)
                if t[i] == 0:
                    miss_tokens += 1
            x.append(t)
            # x.append([vocab2id.get(w, 0) for w in words] + [0]*(max_sent_len-len(words)))
    print ("sentence2id miss_tokens cnt: {}".format(miss_tokens))
    return np.array(x, dtype=np.int32), np.array(y, dtype=np.int32), np.array(seq_len, dtype=np.float32)


def sentence_tfidf_and_pad(in_file, tokens_idf, max_sent_len):
    tfidf_weight = []
    with open(in_file, 'r') as fr:
        for line in fr:
            t = line.strip().split(g_divide)
            words = clean_str(t[1])
            words = words[: max_sent_len]
            # tfidf_weight.append(tokens_idf.get(w, 0) for w in words + [0]*(max_sent_len-len(words)))
            t = [0] * max_sent_len
            for i, w in enumerate(words):
                if i == max_sent_len:
                    break
                t[i] = tokens_idf.get(w, 0)
            tfidf_weight.append(t)
            # print (tfidf_weight[-1])
            # input()
    return np.array(tfidf_weight, dtype=np.float32)


def batch_iter(x, y, seq_len=None, batch_size=None, shuffle=False):
    assert len(x) == len(y)
    idx = np.arange(len(x))
    if shuffle:
        idx = np.random.permutation(len(x))
    if seq_len is not None:
        for start_idx in range(0, len(x), batch_size):
            excerpt = idx[start_idx:start_idx + batch_size]
            yield x[excerpt], y[excerpt], seq_len[excerpt]
    else:
        for start_idx in range(0, len(x), batch_size):
            excerpt = idx[start_idx:start_idx + batch_size]
            yield x[excerpt], y[excerpt]


class DataSet(object):
    def __init__(self, config, load=False):
        self.load = load
        self.train_file = config.train_file
        self.test_file = config.test_file
        self.dev_file = config.dev_file
        self.word_embed_file = config.word_embed_file
        self.vocab2id_file_pkl = config.vocab2id_file_pkl
        self.we_file_pkl = config.we_file_pkl
        self.word_dim = config.word_dim
        self.max_seq_len = config.max_seq_len
        self.idf_file_pkl = config.idf_file_pkl
        self.we = None

        self.train_x = None
        self.train_y = None
        self.train_seq_len = None

        self.test_x = None
        self.test_y = None
        self.test_seq_len = None

        self.dev_x = None
        self.dev_y = None
        self.dev_seq_len = None

        self.init_data()

    def init_data(self):
        if self.load:
            self.we = tools.load_params(self.we_file_pkl)
            vocab2id = tools.load_params(self.vocab2id_file_pkl)
        else:
            self.we, vocab2id = load_embed_from_text(self.word_embed_file, self.word_dim)
            tools.save_params(self.we, self.we_file_pkl)
            tools.save_params(vocab2id, self.vocab2id_file_pkl)
        print("vocab size: %d" % len(vocab2id), "we shape: ", self.we.shape)

        self.train_x, self.train_y, self.train_seq_len = sentence2id_and_pad(
            self.train_file, vocab2id, self.max_seq_len)
        print("train_x: %d " % len(self.train_x), "train_y: %d" % len(self.train_y))

        if self.dev_file is not None:
            self.dev_x, self.dev_y, self.dev_seq_len = sentence2id_and_pad(
                self.dev_file, vocab2id, self.max_seq_len)
            print("dev_x: %d " % len(self.dev_x), "dev_y: %d" % len(self.dev_y))

        if self.test_file is not None:
            self.test_x, self.test_y, self.test_seq_len = sentence2id_and_pad(
                self.test_file, vocab2id, self.max_seq_len)
            print("test_x: %d " % len(self.test_x), "test_y: %d" % len(self.test_y))


class DataSetIDF(object):
    def __init__(self, config, load=False):
        self.load = load
        self.train_file = config.train_file
        self.test_file = config.test_file
        self.dev_file = config.dev_file
        self.word_embed_file = config.word_embed_file
        self.vocab2id_file_pkl = config.vocab2id_file_pkl
        self.we_file_pkl = config.we_file_pkl
        self.idf_file_pkl = config.idf_file_pkl
        self.word_dim = config.word_dim
        self.max_seq_len = config.max_seq_len
        self.we = None

        self.train_x = None
        self.train_y = None
        self.train_seq_len = None
        self.train_tfidf = None

        self.test_x = None
        self.test_y = None
        self.test_seq_len = None
        self.test_tfidf = None

        self.dev_x = None
        self.dev_y = None
        self.dev_seq_len = None
        self.dev_tfidf = None

        self.init_data()

    def init_data(self):
        if self.load:
            self.we = tools.load_params(self.we_file_pkl)
            vocab2id = tools.load_params(self.vocab2id_file_pkl)
            tokens_idf = tools.load_params(self.idf_file_pkl)
        else:
            self.we, vocab2id = load_embed_from_text(self.word_embed_file, self.word_dim)
            tokens_idf = idf([my_config.train_file, my_config.test_file])
            tools.save_params(tokens_idf, self.idf_file_pkl)
            tools.save_params(self.we, self.we_file_pkl)
            tools.save_params(vocab2id, self.vocab2id_file_pkl)
        print("vocab size: %d" % len(vocab2id), "we shape: ", self.we.shape)

        self.train_x, self.train_y, self.train_seq_len = sentence2id_and_pad(
            self.train_file, vocab2id, self.max_seq_len)
        print("train_x: %d " % len(self.train_x), "train_y: %d" % len(self.train_y))
        self.train_tfidf = sentence_tfidf_and_pad(
            self.train_file, tokens_idf, self.max_seq_len)

        if self.dev_file is not None:
            self.dev_x, self.dev_y, self.dev_seq_len = sentence2id_and_pad(
                self.dev_file, vocab2id, self.max_seq_len)
            print("dev_x: %d " % len(self.dev_x), "dev_y: %d" % len(self.dev_y))
            self.dev_tfidf = sentence_tfidf_and_pad(
                self.dev_file, tokens_idf, self.max_seq_len)

        if self.test_file is not None:
            self.test_x, self.test_y, self.test_seq_len = sentence2id_and_pad(
                self.test_file, vocab2id, self.max_seq_len)
            print("test_x: %d " % len(self.test_x), "test_y: %d" % len(self.test_y))
            self.test_tfidf = sentence_tfidf_and_pad(
                self.test_file, tokens_idf, self.max_seq_len)


if __name__ == '__main__':
    from config import ConfigAvgTFIDF as Config
    my_config = Config()

    #tools.get_max_seq_len(my_config.train_file)
    #tools.get_max_seq_len(my_config.test_file)
    #tools.get_class_labels(my_config.train_file, g_divide)
    #tools.get_class_labels(my_config.test_file, g_divide)
    #idf([my_config.train_file, my_config.test_file])
    data = DataSet(my_config)
    data = DataSetIDF(my_config)
    print (data.train_tfidf.shape)
    print(data.test_tfidf.shape)
