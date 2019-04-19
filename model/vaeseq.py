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
from opennmt.utils.parallel import GraphDispatcher
from opennmt import constants
import os
import ipdb
import copy
import yaml
import io

from tensorflow.python.eager import context
from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.layers.base import InputSpec
from tensorflow.python.layers.base import Layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import standard_ops
from tensorflow.python.util.tf_export import tf_export


def create_embeddings(vocab_size, depth=512):
      """Creates an embedding variable."""
      return tf.get_variable("embedding", shape = [vocab_size, depth])

class VAESEQ_Model:

    def _compute_loss(self, outputs, tgt_ids_batch, tgt_length, params, mode, mu_states, logvar_states):
        
        if mode == "Training":
            mode = tf.estimator.ModeKeys.TRAIN            
        else:
            mode = tf.estimator.ModeKeys.EVAL            
          
        if self.Loss_type == "Cross_Entropy":
            if isinstance(outputs, dict):
                logits = outputs["logits"]
                attention = outputs.get("attention")
            else:
                logits = outputs
                attention = None 
                            
            loss, loss_normalizer, loss_token_normalizer = cross_entropy_sequence_loss(
                logits,
                tgt_ids_batch, 
                tgt_length + 1,                                                         
                label_smoothing = params.get("label_smoothing", 0.0),
                average_in_time = params.get("average_loss_in_time", True),
                mode = mode
            )
            
            
            #----- Calculating kl divergence --------

            kld_loss = -0.5 * tf.reduce_sum(logvar_states - tf.pow(mu_states, 2) - tf.exp(logvar_states) + 1, 1)

            return loss, loss_normalizer, loss_token_normalizer, kld_loss
        
    
    def _initializer(self, params):
        
        if params["Architecture"] == "Transformer":
            print("tf.variance_scaling_initializer")
            return tf.variance_scaling_initializer(
        mode="fan_avg", distribution="uniform", dtype=self.dtype)
        else:            
            param_init = params.get("param_init")
            if param_init is not None:
                print("tf.random_uniform_initializer")
                return tf.random_uniform_initializer(
              minval=-param_init, maxval=param_init, dtype=self.dtype)
        return None
        
    def __init__(self, config_file, mode, test_feature_file=None):

        def _normalize_loss(num, den=None):
            """Normalizes the loss."""
            if isinstance(num, list):  # Sharded mode.
                if den is not None:
                    assert isinstance(den, list)
                    return tf.add_n(num) / tf.add_n(den) #tf.reduce_mean([num_/den_ for num_,den_ in zip(num, den)]) #tf.add_n(num) / tf.add_n(den)
                else:
                    return tf.reduce_mean(num)
            elif den is not None:
                return num / den
            else:
                return num

        def _extract_loss(loss, Loss_type="Cross_Entropy"):
            """Extracts and summarizes the loss."""
            losses = None
            print("loss numb:", len(loss))
            if Loss_type=="Cross_Entropy":
                if not isinstance(loss, tuple):                    
                    print(1)
                    actual_loss = _normalize_loss(loss)
                    tboard_loss = actual_loss
                    tf.summary.scalar("loss", tboard_loss)
                    losses = actual_loss                    
                else:                         
                    actual_loss = _normalize_loss(loss[0], den=loss[1])
                    tboard_loss = _normalize_loss(loss[0], den=loss[2]) if len(loss) > 2 else actual_loss
                    tf.summary.scalar("loss", tboard_loss)            
                    losses = actual_loss
                    loss_kd = _normalize_loss(loss[3])

            return losses,loss_kd                         

        def _loss_op(inputs, params, mode):
            """Single callable to compute the loss."""
            logits, _, tgt_ids_out, tgt_length, mu_states, logvar_states  = self._build(inputs, params, mode)
            losses = self._compute_loss(logits, tgt_ids_out, tgt_length, params, mode, mu_states, logvar_states)
            
            return losses

        with open(config_file, "r") as stream:
            config = yaml.load(stream)

        Loss_type = config.get("Loss_Function","Cross_Entropy")
        
        self.Loss_type = Loss_type
        self.config = config 
        self.using_tf_idf = config.get("using_tf_idf", False)
        
        train_batch_size = config["training_batch_size"]   
        eval_batch_size = config["eval_batch_size"]
        
        self.latent_variable_size = config.get("latent_variable_size",128)
        
        max_len = config["max_len"]
        
        example_sampling_distribution = config.get("example_sampling_distribution",None)
        self.dtype = tf.float32
        
        # Input pipeline:
        # Return lookup table of type index_table_from_file
        src_vocab, _ = load_vocab(config["src_vocab_path"], config["src_vocab_size"])
        tgt_vocab, _ = load_vocab(config["tgt_vocab_path"], config["tgt_vocab_size"])
        
        load_data_version = config.get("dataprocess_version",None)
        
        if mode == "Training":    
            print("num_devices", config.get("num_devices",1))
            
            dispatcher = GraphDispatcher(
                config.get("num_devices",1), 
                daisy_chain_variables=config.get("daisy_chain_variables",False), 
                devices= config.get("devices",None)
            ) 
            
            batch_multiplier = config.get("num_devices", 1)
            num_threads = config.get("num_threads", 4)
            
            if Loss_type == "Wasserstein":
                self.using_tf_idf = True
                
            if self.using_tf_idf:
                tf_idf_table = build_tf_idf_table(
                    config["tgt_vocab_path"], 
                    config["tgt_vocab_size"], 
                    config["domain_numb"], 
                    config["training_feature_file"])           
                self.tf_idf_table = tf_idf_table
                
            iterator = load_data(
                config["training_label_file"], 
                src_vocab, 
                batch_size = train_batch_size, 
                batch_type=config["training_batch_type"], 
                batch_multiplier = batch_multiplier, 
                tgt_path=config["training_feature_file"], 
                tgt_vocab=tgt_vocab, 
                max_len = max_len, 
                mode=mode, 
                shuffle_buffer_size = config["sample_buffer_size"], 
                num_threads = num_threads, 
                version = load_data_version, 
                distribution = example_sampling_distribution
            )
            
            inputs = iterator.get_next()
            data_shards = dispatcher.shard(inputs)

            with tf.variable_scope(config["Architecture"], initializer=self._initializer(config)):
                losses_shards = dispatcher(_loss_op, data_shards, config, mode)

            self.loss = _extract_loss(losses_shards, Loss_type=Loss_type) 

        elif mode == "Inference": 
            assert test_feature_file != None
            
            iterator = load_data(
                test_feature_file, 
                src_vocab, 
                batch_size = eval_batch_size, 
                batch_type = "examples", 
                batch_multiplier = 1, 
                max_len = max_len, 
                mode = mode, 
                version = load_data_version
            )
            
            inputs = iterator.get_next() 
            
            with tf.variable_scope(config["Architecture"]):
                _ , self.predictions, _, _, _, _ = self._build(inputs, config, mode)
            
        self.iterator = iterator
        self.inputs = inputs
        
    def loss_(self):
        return self.loss
    
    def prediction_(self):
        return self.predictions
   
    def inputs_(self):
        return self.inputs
    
    def iterator_initializers(self):
        if isinstance(self.iterator,list):
            return [iterator.initializer for iterator in self.iterator]
        else:
            return [self.iterator.initializer]        
           
    def _build(self, inputs, config, mode):        

        debugging = config.get("debugging", False)
        Loss_type = self.Loss_type       
        print("Loss_type: ", Loss_type)           

        hidden_size = config["hidden_size"]       
        print("hidden size: ", hidden_size)
                
        tgt_vocab_rev = tf.contrib.lookup.index_to_string_table_from_file(config["tgt_vocab_path"], vocab_size= int(config["tgt_vocab_size"]) - 1, default_value=constants.UNKNOWN_TOKEN)
        
        end_token = constants.END_OF_SENTENCE_ID
        
        # Embedding        
        size_src = config.get("src_embedding_size",512)
        size_tgt = config.get("tgt_embedding_size",512)
        latent_variable_size = self.latent_variable_size
        
        with tf.variable_scope("src_embedding"):
            src_emb = create_embeddings(config["src_vocab_size"], depth=size_src)

        with tf.variable_scope("tgt_embedding"):
            tgt_emb = create_embeddings(config["tgt_vocab_size"], depth=size_tgt)

        self.tgt_emb = tgt_emb
        self.src_emb = src_emb

        # Build encoder, decoder
    #---------------------------------------------GRU-----------------------------------------#
        if config["Architecture"] == "GRU":
            nlayers = config.get("nlayers",4)
            
    #==============================ENCODER==================================
            encoder = onmt.encoders.BidirectionalRNNEncoder(
                nlayers, 
                hidden_size, 
                reducer=onmt.layers.ConcatReducer(), 
                cell_class = tf.contrib.rnn.GRUCell, 
                dropout=0.1, 
                residual_connections=True
            )
    #==============================DECODER==================================
            decoder = onmt.decoders.AttentionalRNNDecoder(
                nlayers, 
                hidden_size, 
                bridge=onmt.layers.CopyBridge(), 
                cell_class=tf.contrib.rnn.GRUCell, 
                dropout=0.1, 
                residual_connections=True
            )
    
    #---------------------------------------------LSTM-----------------------------------------# 
        elif config["Architecture"] == "LSTM":
            nlayers = config.get("nlayers",4)

    #==============================ENCODER==================================
            encoder = onmt.encoders.BidirectionalRNNEncoder(
                nlayers, 
                num_units=hidden_size, 
                reducer=onmt.layers.ConcatReducer(), 
                cell_class=tf.nn.rnn_cell.LSTMCell,
                dropout=0.1, 
                residual_connections=True
            )
    #==============================DECODER==================================
            decoder = onmt.decoders.AttentionalRNNDecoder(
                nlayers, 
                num_units=hidden_size, 
                bridge=onmt.layers.CopyBridge(), 
                attention_mechanism_class=tf.contrib.seq2seq.LuongAttention,
                cell_class=tf.nn.rnn_cell.LSTMCell, 
                dropout=0.1, 
                residual_connections=True
            )
    
    #---------------------------------------------TRANSFORMER-----------------------------------------#
        elif config["Architecture"] == "Transformer":
            nlayers = config.get("nlayers",6)

    #==============================ENCODER==================================
    # Requires 2 encoder for SVAE
            encoder_src = onmt.encoders.self_attention_encoder.SelfAttentionEncoder(
                nlayers, 
                num_units=hidden_size, 
                num_heads=8, 
                ffn_inner_dim=2048, 
                dropout=0.1, 
                attention_dropout=0.1, 
                relu_dropout=0.1
            )  
            
            encoder_tgt = onmt.encoders.self_attention_encoder.SelfAttentionEncoder(
                nlayers, 
                num_units=hidden_size, 
                num_heads=8, 
                ffn_inner_dim=2048, 
                dropout=0.1, 
                attention_dropout=0.1, 
                relu_dropout=0.1
            )
    #==============================DECODER==================================
            decoder = onmt.decoders.self_attention_decoder.SelfAttentionDecoder(
                nlayers, 
                num_units=hidden_size, 
                num_heads=8, 
                ffn_inner_dim=2048, 
                dropout=0.1, 
                attention_dropout=0.1, 
                relu_dropout=0.1
            )
        print("Model type: ", config["Architecture"])
        
        assert config["Architecture"] == "Transformer", \
            "We have only try with Transfomer !"
        
        output_layer = None

    #-----------------------------------------------------TRAINING MODE-----------------------------------------------#
        if mode =="Training":            
            print("Building model in Training mode")
            
            src_length = inputs["src_length"]
            tgt_length = inputs["tgt_length"]
            
            emb_src_batch = tf.nn.embedding_lookup(src_emb, inputs["src_ids"]) # dim = [batch, length, depth]
            emb_tgt_batch = tf.nn.embedding_lookup(tgt_emb, inputs["tgt_ids"])  
            emb_tgt_batch_in = tf.nn.embedding_lookup(tgt_emb, inputs["tgt_ids_in"])   
            
            self.emb_tgt_batch = emb_tgt_batch
            self.emb_src_batch = emb_src_batch
            
            print("emb_src_batch: ", emb_src_batch)
            print("emb_tgt_batch: ", emb_tgt_batch)
            
            tgt_ids_batch = inputs["tgt_ids_out"]
            
            
            #========ENCODER_PROCESS======================
            with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
                
                encoder_output_src = encoder_src.encode(
                    emb_src_batch, 
                    sequence_length = src_length, 
                    mode=tf.estimator.ModeKeys.TRAIN
                )
                
                encoder_output_tgt = encoder_tgt.encode(
                    emb_tgt_batch, 
                    sequence_length = tgt_length, 
                    mode=tf.estimator.ModeKeys.TRAIN
                )
                
                encoder_outputs_src, encoder_states_src, encoder_seq_length_src = encoder_output_src
                encoder_outputs_tgt, encoder_states_tgt, _ = encoder_output_tgt
                    # encoder_outputs: [batch_size, max_length, hidden_units_size]
                    # encoder_states: nlayers ° [batch_size, hidden_units_size] (tuple of 2d array)
                    
                encoder_states_combined = tf.concat([encoder_states_src[-1], encoder_states_tgt[-1]], 1)
                    # dim = [batch_size, hidden_size + hidden_size]
                    
                # important infos
                batch_size = tf.shape(encoder_outputs_src)[0]
                max_length = tf.shape(encoder_outputs_src)[1]
                    
            #======GENERATIVE_PROCESS====================
            with tf.variable_scope("generator"):

                W_out_to_mu = tf.get_variable('output_to_mu_weight', shape = [2 * hidden_size, latent_variable_size])
                b_out_to_mu = tf.get_variable('output_to_mu_bias', shape = [latent_variable_size])

                mu = tf.nn.sigmoid(tf.add(tf.matmul(encoder_states_combined, W_out_to_mu), b_out_to_mu))

                W_out_to_logvar = tf.get_variable('output_to_logvar_weight', shape = [2 * hidden_size, latent_variable_size])
                b_out_to_logvar = tf.get_variable('output_to_logvar_bias', shape = [latent_variable_size])

                logvar = tf.nn.sigmoid(tf.add(tf.matmul(encoder_states_combined, W_out_to_logvar), b_out_to_logvar))

                std = tf.exp(0.5 * logvar)

                z = tf.random_normal([batch_size, latent_variable_size])
                    # z, mu, logvar shape [batch_size, latent_size]

                z = z * std + mu 

                z = tf.reshape(z, [batch_size,1,-1]) 
                    # shape [batch_size, 1, latent_size]

                zz = tf.tile(z,[1, max_length, 1])
                    # shape [batch_size, max_length, latent_size]

                new_memory_inputs = tf.concat([encoder_outputs_src, zz], 2)

                new_memory_inputs = tf.reshape(new_memory_inputs,[batch_size, -1, hidden_size + latent_variable_size])

                print(encoder_outputs_src)
                print(new_memory_inputs)
                
            #======DECODER_PROCESS====================
            with tf.variable_scope("decoder"): 

                logits, _, _, attention = decoder.decode(
                                          emb_tgt_batch_in, 
                                          tgt_length + 1,
                                          vocab_size = int(config["tgt_vocab_size"]),
                                          initial_state = encoder_states_src,
                                          output_layer = output_layer,                                              
                                          mode = tf.estimator.ModeKeys.TRAIN,
                                          memory = new_memory_inputs,
                                          memory_sequence_length = encoder_seq_length_src,
                                          return_alignment_history = True
                )                     
                outputs = {
                       "logits": logits
                       }
                
            predictions = None

    #-----------------------------------------------------INFERENCE MODE-----------------------------------------------#
        elif mode == "Inference":
            
            print("Build model in Inference mode")
            
            src_length = inputs["src_length"]
            emb_src_batch = tf.nn.embedding_lookup(src_emb, inputs["src_ids"])
            
            
            start_tokens = tf.fill([tf.shape(inputs["src_ids"])[0]], constants.START_OF_SENTENCE_ID)
            
           #========ENCODER_PROCESS======================
            with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
                
                encoder_output_src = encoder_src.encode(
                    emb_src_batch, 
                    sequence_length = src_length, 
                    mode=tf.estimator.ModeKeys.TRAIN
                )
                
                encoder_outputs_src, encoder_states_src, encoder_seq_length_src = encoder_output_src
                
                # important infos
                batch_size = tf.shape(encoder_outputs_src)[0]
                max_length = tf.shape(encoder_outputs_src)[1]
            
            #======GENERATIVE_PROCESS====================
            with tf.variable_scope("generator"):
                
                z = tf.random_normal([batch_size, latent_variable_size])
                    # z, mu, logvar shape [batch_size, latent_size]

                z = tf.reshape(z, [batch_size,1,-1]) 
                    # shape [batch_size, 1, latent_size]

                zz = tf.tile(z,[1, max_length, 1])
                    # shape [batch_size, max_length, latent_size]

                new_memory_inputs = tf.concat([encoder_outputs_src, zz], 2)

                new_memory_inputs = tf.reshape(new_memory_inputs,[batch_size, -1, hidden_size + latent_variable_size])
            
            #======DECODER_PROCESS====================
            with tf.variable_scope("decoder"):        
                beam_width = config.get("beam_width", 5)
                print("Inference with beam width %d"%(beam_width))
                maximum_iterations = config.get("maximum_iterations", tf.round(2 * max_length))
               
                if beam_width <= 1:                
                    sampled_ids, _, sampled_length, log_probs, alignment = decoder.dynamic_decode(
                                                                    tgt_emb,
                                                                    start_tokens,
                                                                    end_token,
                                                                    vocab_size=int(config["tgt_vocab_size"]),
                                                                    initial_state=encoder_states_src,
                                                                    maximum_iterations=maximum_iterations,
                                                                    output_layer = output_layer,
                                                                    mode=tf.estimator.ModeKeys.PREDICT,
                                                                    memory=new_memory_inputs,
                                                                    memory_sequence_length=encoder_seq_length_src,
                                                                    dtype=tf.float32,
                                                                    return_alignment_history=True
                                                                    )
                else:
                    length_penalty = config.get("length_penalty", 0)
                    sampled_ids, _, sampled_length, log_probs, alignment = decoder.dynamic_decode_and_search(
                                                          tgt_emb,
                                                          start_tokens,
                                                          end_token,
                                                          vocab_size = int(config["tgt_vocab_size"]),
                                                          initial_state = encoder_states_src,
                                                          beam_width = beam_width,
                                                          length_penalty = length_penalty,
                                                          maximum_iterations = maximum_iterations,
                                                          output_layer = output_layer,
                                                          mode = tf.estimator.ModeKeys.PREDICT,
                                                          memory = new_memory_inputs,
                                                          memory_sequence_length = encoder_seq_length_src,
                                                          dtype=tf.float32,
                                                          return_alignment_history = True)
                    
            target_tokens = tgt_vocab_rev.lookup(tf.cast(sampled_ids, tf.int64))
            
            predictions = {
              "tokens": target_tokens,
              "length": sampled_length,
              "log_probs": log_probs,
              "alignment": alignment,
            }
            
            tgt_ids_batch = None
            tgt_length = None
            mu = None
            logvar = None
            outputs = None
            
        self.outputs = outputs
        
        return outputs, predictions, tgt_ids_batch, tgt_length, mu, logvar         