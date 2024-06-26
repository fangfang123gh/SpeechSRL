# -*- coding: utf-8 -*-

import argparse
from supar import VISemanticRoleLabelingParser
from supar.cmds.cmd import parse


def main():
    parser = argparse.ArgumentParser(description='Create SRL using Variational Inference.')
    parser.set_defaults(Parser=VISemanticRoleLabelingParser)
    subparsers = parser.add_subparsers(title='Commands', dest='mode')
    # train
    subparser = subparsers.add_parser('train', help='Train a parser.')
    subparser.add_argument('--feat', '-f', choices=['tag', 'char', 'lemma', 'bert'], nargs='+', help='features to use')
    subparser.add_argument('--build', '-b', action='store_true', help='whether to build the model first')
    subparser.add_argument('--encoder', choices=['lstm', 'bert'], default='lstm', help='encoder to use')
    subparser.add_argument('--max-len', type=int, help='max length of the sentences')
    subparser.add_argument('--buckets', default=32, type=int, help='max num of buckets to use')
    subparser.add_argument('--train', default='data/sdp/DM/train.conllu', help='path to train file')
    subparser.add_argument('--dev', default='data/sdp/DM/dev.conllu', help='path to dev file')
    subparser.add_argument('--test', default='data/sdp/DM/test.conllu', help='path to test file')
    subparser.add_argument('--embed', default='data/glove.6B.100d.txt', help='path to pretrained embeddings')
    subparser.add_argument('--unk', default='unk', help='unk token in pretrained embeddings')
    subparser.add_argument('--n-embed', default=100, type=int, help='dimension of embeddings')
    subparser.add_argument('--n-embed-proj', default=125, type=int, help='dimension of projected embeddings')
    subparser.add_argument('--bert', default='bert-base-cased', help='which bert model to use')
    subparser.add_argument('--inference', default='mfvi', choices=['mfvi', 'lbp'], help='approximate inference methods')
    subparser.add_argument('--lr_rate', default=1, type=int)
    subparser.add_argument('--patience', default=3, type=int)
    
    subparser.add_argument('--split',
                           action='store_true',
                           help='whether to use different mlp for predicate and arg')
    subparser.add_argument('--train_given_prd',
                           action='store_true',
                           default=False,
                           help='whether use predicate embedding')
    # evaluate
    subparser = subparsers.add_parser('evaluate', help='Evaluate the specified parser and dataset.')
    subparser.add_argument('--buckets', default=8, type=int, help='max num of buckets to use')
    subparser.add_argument('--data', default='data/sdp/DM/test.conllu', help='path to dataset')
    subparser.add_argument('--given_prd', action='store_true', default=False)
    # predict
    subparser = subparsers.add_parser('predict', help='Use a trained parser to make predictions.')
    subparser.add_argument('--buckets', default=8, type=int, help='max num of buckets to use')
    subparser.add_argument('--data', default='data/sdp/DM/test.conllu', help='path to dataset')
    subparser.add_argument('--pred', default='pred.conllu', help='path to predicted result')
    subparser.add_argument('--prob', action='store_true', help='whether to output probs')
    subparser.add_argument('--task', choices=['05', '09', '12'], required=True, help='which dataset')
    subparser.add_argument('--gold',
                           default='data/conll05-original-style/sc-wsj.final')
    subparser.add_argument('--vtb',
                           action='store_true',
                           default=False,
                           help='whether to use viterbi')
    subparser.add_argument('--given_prd',
                           action='store_true',
                           default=False,
                           help='whether to use given predicates to constrain')

    # api
    subparser = subparsers.add_parser('api', help='test api for cup')
    subparser.add_argument('--task', default='09', help='which dataset')
    subparser.add_argument('--buckets', default=8, type=int, help='max num of buckets to use')
    subparser.add_argument('--input', default='中国 建筑业 对 外 开放 始于 八十年代 。')
    subparser.add_argument('--given_prd', action='store_true', default=False)
    parse(parser)


if __name__ == "__main__":
    main()
