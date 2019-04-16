import tensorflow as tf
import opennmt as onmt
from .dataprocess import *
import argparse
import sys
import numpy as np
from opennmt.inputters.text_inputter import load_pretrained_embeddings
from opennmt.utils.losses import cross_entropy_sequence_loss
from opennmt.utils.evaluator import *
import model 
import os
import ipdb
import yaml
import io
import sklearn.metrics as sk
np.set_printoptions(threshold=sys.maxsize)

def inference(config_file, checkpoint_path=None, test_feature_file=None):
    
    with open(config_file, "r") as stream:
        config = yaml.load(stream)
    assert test_feature_file!=None 
    from opennmt.utils.misc import print_bytes
    graph = tf.Graph()    
    with tf.Session(graph=graph,config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True))) as sess_:
     
        eval_model = model.Model(config_file, "Inference", test_feature_file)
        #emb_src_batch = eval_model.emb_src_batch_()
        saver = tf.train.Saver()
        tf.tables_initializer().run()
        tf.global_variables_initializer().run()

        if checkpoint_path==None:
            checkpoint_dir = config["model_dir"]
            checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)

        print(("Evaluating model %s"%checkpoint_path))
        saver.restore(sess_, checkpoint_path)        

        predictions = eval_model.prediction_()
        tokens = predictions["tokens"]
        length = predictions["length"]            
        
        sess_.run(eval_model.iterator_initializers())
        print("write to :%s"%os.path.join(config["model_dir"],"eval",os.path.basename(test_feature_file) + ".trans." + os.path.basename(checkpoint_path)))
        with open(os.path.join(config["model_dir"],"eval",os.path.basename(test_feature_file) + ".trans." + os.path.basename(checkpoint_path)),"w") as output_:
            while True:                 
                try:                
                    _tokens, _length = sess_.run([tokens, length])                    
                    #print emb_src_batch_
                    for b in range(_tokens.shape[0]):                        
                        pred_toks = _tokens[b][0][:_length[b][0] - 1]                                                
                        pred_sent = b" ".join(pred_toks)                        
                        print_bytes(pred_sent, output_)                                            
                except tf.errors.OutOfRangeError:
                    break
        
    return os.path.join(config["model_dir"],"eval",os.path.basename(test_feature_file) + ".trans." + os.path.basename(checkpoint_path))
