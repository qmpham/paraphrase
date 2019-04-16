import tensorflow as tf
import opennmt as onmt
from utils.dataprocess import *
from utils.utils_ import *
import argparse
import sys
import numpy as np
from opennmt.inputters.text_inputter import load_pretrained_embeddings
from opennmt.utils.losses import cross_entropy_sequence_loss
from opennmt.utils.evaluator import *
from model import *
import os
import ipdb
import yaml
import io
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--config_file", required=True , help="configuration file")
parser.add_argument("--eval_step", required=True , help="evaluation step")
args = parser.parse_args()

config_file = args.config_file
with open(config_file, "r") as stream:
    config = yaml.load(stream)

"""
test_feature_files = ["/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/test_corpora/ecb/ECB.en-fr.en.test.bpe.tok","/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/test_corpora/emea/khresmoi-summary-test.en.bpe.tok","/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/test_corpora/epps/test2007-enfr.en.txt.bpe.tok","/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/test_corpora/news/newstest2009-enfr.en.txt.bpe.tok"]

test_label_files = ["/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/test_corpora/ecb/ECB.en-fr.fr.test.bpe.tok","/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/test_corpora/emea/khresmoi-summary-test.fr.bpe.tok","/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/test_corpora/epps/test2007-enfr.fr.txt.bpe.tok","/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/test_corpora/news/newstest2009-enfr.fr.txt.bpe.tok"]

test_tag_files = ["/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/test_corpora/ecb/ECB.en-fr.en.test.bpe.tok.tag.0","/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/test_corpora/emea/khresmoi-summary-test.en.bpe.tok.tag.0","/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/test_corpora/epps/test2007-enfr.en.txt.bpe.tok.tag.0","/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/test_corpora/news/newstest2009-enfr.en.txt.bpe.tok.tag.0"]
"""
"""
test_feature_files = ["/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/test_corpora/ecb/ECB.en-fr.en.test.bpe.tok","/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/test_corpora/epps/test2007-enfr.en.txt.bpe.tok","/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/test_corpora/ted/IWSLT16.TED.tst2010.en-fr.en.bpe.tok"]

test_label_files = ["/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/test_corpora/ecb/ECB.en-fr.fr.test.bpe.tok","/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/test_corpora/epps/test2007-enfr.fr.txt.bpe.tok","/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/test_corpora/ted/IWSLT16.TED.tst2010.en-fr.fr.bpe.tok"]

test_tag_files = ["/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/test_corpora/ecb/ECB.en-fr.en.test.bpe.tok.tag.1","/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/test_corpora/epps/test2007-enfr.en.txt.bpe.tok.tag.1","/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/test_corpora/ted/IWSLT16.TED.tst2010.en-fr.en.bpe.tok.tag.1"]
"""
"""
test_feature_files = ["/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/test_corpora/emea/EMEA.en-fr.en.bpe.tok.tst","/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/test_corpora/epps/europarl-v7.fr-en.en.bpe.tok.tst.o","/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/test_corpora/news/news-commentary-v12.fr-en.en.bpe.tok.tst","/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/test_corpora/ecb/ECB.en-fr.en.test.bpe.tok"]

test_label_files = ["/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/test_corpora/emea/EMEA.en-fr.fr.bpe.tok.tst","/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/test_corpora/epps/europarl-v7.fr-en.fr.bpe.tok.tst","/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/test_corpora/news/news-commentary-v12.fr-en.fr.bpe.tok.tst","/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/test_corpora/ecb/ECB.en-fr.fr.test.bpe.tok"]

test_tag_files = ["/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/test_corpora/emea/EMEA.en-fr.en.bpe.tok.tst.tag.1","/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/test_corpora/epps/europarl-v7.fr-en.en.bpe.tok.tst.o.tag.1","/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/test_corpora/news/news-commentary-v12.fr-en.en.bpe.tok.tst.tag.1","/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/test_corpora/ecb/ECB.en-fr.en.test.bpe.tok.tag.0"]
"""
test_feature_files = ["/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/test_corpora/emea/EMEA.en-fr.en.bpe.tok.tst","/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/test_corpora/epps/europarl-v7.fr-en.en.bpe.tok.tst.o","/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/test_corpora/ecb/ECB.en-fr.en.test.bpe.tok","/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/test_corpora/emea/khresmoi-summary-test.en.bpe.tok","/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/test_corpora/epps/test2007-enfr.en.txt.bpe.tok","/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/test_corpora/news/newstest2009-enfr.en.txt.bpe.tok","/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/test_corpora/ted/IWSLT16.TED.tst2010.en-fr.en.bpe.tok"]

test_label_files = ["/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/test_corpora/emea/EMEA.en-fr.fr.bpe.tok.tst","/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/test_corpora/epps/europarl-v7.fr-en.fr.bpe.tok.tst.o","/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/test_corpora/ecb/ECB.en-fr.fr.test.bpe.tok","/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/test_corpora/emea/khresmoi-summary-test.fr.bpe.tok","/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/test_corpora/epps/test2007-enfr.fr.txt.bpe.tok","/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/test_corpora/news/newstest2009-enfr.fr.txt.bpe.tok","/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/test_corpora/ted/IWSLT16.TED.tst2010.en-fr.fr.bpe.tok"]

test_tag_files = ["/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/test_corpora/emea/EMEA.en-fr.en.bpe.tok.tst.tag.1","/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/test_corpora/epps/europarl-v7.fr-en.en.bpe.tok.tst.o.tag.1","/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/test_corpora/ecb/ECB.en-fr.en.test.bpe.tok.tag.1","/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/test_corpora/emea/khresmoi-summary-test.en.bpe.tok.tag.0","/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/test_corpora/epps/test2007-enfr.en.txt.bpe.tok.tag.1","/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/test_corpora/news/newstest2009-enfr.en.txt.bpe.tok.tag.2","/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/test_corpora/ted/IWSLT16.TED.tst2010.en-fr.en.bpe.tok.tag.2"]

print("number of testsets: ", len(test_label_files))
external_evaluator = [None] * len(test_feature_files)

for i in range(len(test_label_files)):
    external_evaluator[i] = BLEUEvaluator(test_label_files[i], config["model_dir"])

with tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
    for i in range(len(test_feature_files)):
        checkpoint_path = os.path.join(config["model_dir"], "model.ckpt-%d"%int(args.eval_step))
        prediction_file = inference(config_file, checkpoint_path, test_feature_files[i])
        score = external_evaluator[i].score(test_label_files[i], prediction_file)
        print("BLEU at checkpoint %s for testset %s: %f"%(checkpoint_path, test_label_files[i], score))
