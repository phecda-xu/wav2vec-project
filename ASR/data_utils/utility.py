# coding:utf-8
#
# Date : 2020.03.21
# Author: phecda-xu < >
#
# DEC:
#     function

import os
import json
import codecs
import tarfile
import hashlib
import numpy as np


def read_manifest(manifest_path, max_duration=float('inf'), min_duration=0.0):
    """
    Load data from manifest file.
    :param manifest_path:  str, path of manifest file
    :param max_duration:   float, max duration of input data
    :param min_duration:   float, min duration of input data
    :return manifest:      list, json list
    """
    manifest = []
    for json_line in codecs.open(manifest_path, 'r', 'utf-8'):
        try:
            json_data = json.loads(json_line)
        except Exception as e:
            raise IOError("Error reading manifest: %s" % str(e))
        if max_duration >= json_data["duration"] >= min_duration:
            manifest.append(json_data)
    return manifest


def sparse_tuple_from(sequences, dtype=np.int32):
    """Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []
    for n, seq in enumerate(sequences):
        indices.extend(zip([n]*len(seq), range(len(seq))))
        values.extend(seq)
    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1]+1], dtype=np.int64)
    return indices, values, shape


def decode_sparse_tensor(sparse_tensor, vocab_list):
    decoded_indexes = list()
    current_i = 0
    current_seq = []

    for offset, i_and_index in enumerate(sparse_tensor[0]):
        i = i_and_index[0]
        if i != current_i:
            decoded_indexes.append(current_seq)
            current_i = i
            current_seq = list()
            current_seq.append(offset)
        decoded_indexes.append(current_seq)

    result = []
    for index in decoded_indexes:
        result.append(decode_a_seq(index, sparse_tensor, vocab_list))
    return result


# 序列编码转换
def decode_a_seq(indexes, spars_tensor, vocab_list):
    decoded = []
    for m in indexes:
        str = vocab_list[spars_tensor[1][m]]
        decoded.append(str)
    return decoded


def md5file(fname):
    hash_md5 = hashlib.md5()
    f = open(fname, "rb")
    for chunk in iter(lambda: f.read(4096), b""):
        hash_md5.update(chunk)
    f.close()
    return hash_md5.hexdigest()


def download(url, md5sum, target_dir):
    """Download file from url to target_dir, and check md5sum."""
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    filepath = os.path.join(target_dir, url.split("/")[-1])
    if not (os.path.exists(filepath) and md5file(filepath) == md5sum):
        print("Downloading %s ..." % url)
        os.system("wget -c " + url + " -P " + target_dir)
        # print("\nMD5 Chesksum %s ..." % filepath)
        # if not md5file(filepath) == md5sum:
        #     raise RuntimeError("MD5 checksum failed.")
    else:
        print("File exists, skip downloading. (%s)" % filepath)
    return filepath


def unpack(filepath, target_dir, rm_tar=False):
    """Unpack the file to the target_dir."""
    print("Unpacking %s ..." % filepath)
    tar = tarfile.open(filepath)
    tar.extractall(target_dir)
    tar.close()
    if rm_tar == True:
        os.remove(filepath)


def pinyin_cover(char):
    if 'zh' in char:
        char = char.replace("zh", "z")
    char = char.replace("z", "z-zh")
    if 'ch' in char:
        char = char.replace("ch", "c")
    char = char.replace("c", "c-ch")
    if 'sh' in char:
        char = char.replace("sh", "s")
    char = char.replace("s", "s-sh")
    if 'l' in char:
        char = char.replace("l", "n")
    if 'ing' in char:
        char = char.replace("ing", "in")
    char = char.replace("in", "in-ing")
    return char