# coding: utf-8
from __future__ import print_function
import numpy as np
import pickle
import os


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


def batch_iter2(x, y, seq_len, tfidf, batch_size=None, shuffle=False):
    assert len(x) == len(y)
    idx = np.arange(len(x))
    if shuffle:
        idx = np.random.permutation(len(x))

    for start_idx in range(0, len(x), batch_size):
        excerpt = idx[start_idx:start_idx + batch_size]
        yield x[excerpt], y[excerpt], seq_len[excerpt], tfidf[excerpt]


def save_params(params, fname):
    if os.path.exists(fname):
        os.remove(fname)
    with open(fname, 'wb') as fw:
        pickle.dump(params, fw, protocol=pickle.HIGHEST_PROTOCOL)


def load_params(fname):
    if not os.path.exists(fname):
        raise RuntimeError('no file: %s' % fname)
    with open(fname, 'rb') as fr:
        params = pickle.load(fr)
    return params


def get_max_seq_len(in_file, divide="\t"):
    max_len = 0
    with open(in_file, 'r') as fr:
        for line in fr:
            t = line.strip().split(divide)
            seq_len = len(t[1].split())
            if seq_len > max_len:
                max_len = seq_len
                print (t[1])
    print ("max_len: {}".format(max_len))


def get_class_labels(in_file, divide):
    labels = set([])
    with open(in_file, 'r') as fr:
        for line in fr:
            t = line.strip().split(divide)
            labels.add(t[0])
    print ("labels: ")
    print (labels)
