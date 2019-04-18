import tensorflow as tf
import opennmt as onmt
from opennmt.utils.optim import *
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
from tensorflow.python.framework import ops
import datetime

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--config_file", required=True , help="configuration file")
args = parser.parse_args()

config_file = args.config_file
with open(config_file, "r") as stream:
    config = yaml.load(stream)
# Eval directory stores prediction files
if not os.path.exists(os.path.join(config["model_dir"],"eval")):
    os.makedirs(os.path.join(config["model_dir"],"eval"))

training_model = Model(config_file, "Training")
global_step = tf.train.create_global_step()

if config.get("Loss_Function","Cross_Entropy")=="Cross_Entropy":
    loss, kl_loss = training_model.loss_()
    kl_weight = kl_coeff(global_step)
    generator_total_loss = loss + kl_loss * kl_weight

inputs = training_model.inputs_()

print(tf.get_default_graph().get_all_collection_keys())

if config["mode"] == "Training":
    optimizer_params = config["optimizer_parameters"]
    with tf.variable_scope("main_training"):
        train_op, accum_vars_ = optimize_loss(generator_total_loss, config["optimizer_parameters"])
    
Eval_dataset_numb = len(config["eval_label_file"])
print("Number of validation set: ", Eval_dataset_numb)
external_evaluator = [None] * Eval_dataset_numb 
writer_bleu = [None] * Eval_dataset_numb 

for i in range(Eval_dataset_numb):
    external_evaluator[i] = BLEUEvaluator(config["eval_label_file"][i], config["model_dir"])
    writer_bleu[i] = tf.summary.FileWriter(os.path.join(config["model_dir"],"BLEU","domain_%d"%i))

with tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
    
    writer = tf.summary.FileWriter(config["model_dir"])    
    var_list_ = tf.global_variables()
    for v in tf.trainable_variables():
        if v not in tf.global_variables():
            var_list_.append(v)
    for v in var_list_:
        print(v.name)
    saver = tf.train.Saver(var_list_, max_to_keep=config["max_to_keep"])
    checkpoint_path = tf.train.latest_checkpoint(config["model_dir"])
        
    sess.run([v.initializer for v in var_list_])    

    sess.run([v.initializer for v in accum_vars_])
    
    training_summary = tf.summary.merge_all()
    global_step_ = sess.run(global_step) 
    if checkpoint_path:
        print("Continue training:...")
        print("Load parameters from %s"%checkpoint_path)
        saver.restore(sess, checkpoint_path)        
        global_step_ = sess.run(global_step)
        print("global_step: ", global_step_)
                            
        for i in range(Eval_dataset_numb):
            prediction_file = inference(config_file, checkpoint_path, config["eval_feature_file"][i])
            score = external_evaluator[i].score(config["eval_label_file"][i], prediction_file)
            print("BLEU at checkpoint %s for testset %s: %f"%(checkpoint_path, config["eval_feature_file"][i], score))            
                
    else:
        print("Training from scratch")
        
    tf.tables_initializer().run()    
    sess.run(training_model.iterator_initializers())
    total_loss = []            
   
    while global_step_ <= config["iteration_number"]:                       

        loss_, global_step_, _ = sess.run([generator_total_loss, global_step, train_op])               
        total_loss.append(loss_)
        
        if (np.mod(global_step_, config["printing_freq"])) == 0:            
            print((datetime.datetime.now()))
            print(("Loss at step %d"%(global_step_), np.mean(total_loss)))                
            
        if (np.mod(global_step_, config["summary_freq"])) == 0:
            training_summary_ = sess.run(training_summary)
            writer.add_summary(training_summary_, global_step=global_step_)
            writer.flush()
            total_loss = []
            
        if (np.mod(global_step_, config["save_freq"])) == 0 and global_step_ > 0:    
            print((datetime.datetime.now()))
            checkpoint_path = os.path.join(config["model_dir"], 'model.ckpt')
            print(("save to %s"%(checkpoint_path)))
            saver.save(sess, checkpoint_path, global_step = global_step_)
                                                                                                                 
        if (np.mod(global_step_, config["eval_freq"])) == 0 and global_step_ >0: 
            checkpoint_path = tf.train.latest_checkpoint(config["model_dir"])
            for i in range(Eval_dataset_numb):
                prediction_file = inference(config_file, checkpoint_path, config["eval_feature_file"][i])
                score = external_evaluator[i].score(config["eval_label_file"][i], prediction_file)
                print("BLEU at checkpoint %s for testset %s: %f"%(checkpoint_path,config["eval_label_file"][i], score))
                score_summary = tf.Summary(value=[tf.Summary.Value(tag="eval_score_%d"%i, simple_value=score)])
                writer_bleu[i].add_summary(score_summary, global_step_)
                writer_bleu[i].flush()