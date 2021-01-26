# coding:utf-8
#
# Date : 2020.04.01
# Author: phecda-xu < >
#
# DEC:
#     build_vocab
import sys
sys.path.append('ASR/')
import codecs
import argparse

from collections import Counter
from data_utils.utility import read_manifest


def count_manifest(counter, manifest_path, word_form='sinogram'):
    manifest_jsons = read_manifest(manifest_path)
    for line_json in manifest_jsons:
        if word_form == 'sinogram':
            for char in line_json['text']:
                counter.update(char)
        elif word_form == 'english':
            for char in line_json['fully_pinyin']:
                counter.update(char)
        elif word_form == 'pinyin':
            counter.update(line_json['pinyin'].split(' '))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest_paths',
                        type=str,
                        default=None,
                        nargs='+',
                        help="path to train data set")
    parser.add_argument('--count_threshold',
                        type=int,
                        default=0,
                        help="Truncation threshold for char counts.")
    parser.add_argument('--vocab_path',
                        type=str,
                        default='Work/aishell_1/data/vocab.txt',
                        help="Filepath to write the vocabulary.")
    parser.add_argument('--word_form',
                        type=str,
                        default='sinogram',
                        help="sinogram or pinyin.")

    params = parser.parse_args()
    #
    print("***************************************")
    print("args info:")
    for param, value in params.__dict__.items():
        print("{} = {}".format(param, value))
    print("***************************************")
    #

    counter = Counter()
    counter.update('_')
    counter.subtract('_')
    for manifest_path in params.manifest_paths:
        count_manifest(counter, manifest_path, params.word_form)

    count_sorted = sorted(counter.items(), key=lambda x: x[1], reverse=False)
    with codecs.open(params.vocab_path, 'w', 'utf-8') as fout:
        for char, count in count_sorted:
            if count < params.count_threshold:
                break
            fout.write(char + '\n')


if __name__ == '__main__':
    main()
