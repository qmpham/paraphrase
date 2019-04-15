import tensorflow as tf
import opennmt as onmt
from opennmt import constants
from opennmt.utils.misc import count_lines
import numpy as np
from collections import OrderedDict

def random_shard(shard_size, dataset_size):    
    num_shards = -(-dataset_size // shard_size)  # Ceil division.
    offsets = np.linspace(0, dataset_size, num=num_shards, endpoint=False, dtype=np.int64)

    def _random_shard(dataset):
        sharded_dataset = tf.data.Dataset.from_tensor_slices(offsets)
        sharded_dataset = sharded_dataset.shuffle(num_shards)
        sharded_dataset = sharded_dataset.flat_map(
            lambda offset: dataset.skip(offset).take(shard_size))
        return sharded_dataset

    return _random_shard

def get_dataset_size(data_file):
    return count_lines(data_file)

def get_padded_shapes(dataset):    
    return tf.contrib.framework.nest.map_structure(
    lambda shape: shape.as_list(), dataset.output_shapes)

def filter_irregular_batches(multiple):    
    if multiple == 1:
        return lambda dataset: dataset

    def _predicate(*x):
        flat = tf.contrib.framework.nest.flatten(x)
        batch_size = tf.shape(flat[0])[0]
        return tf.equal(tf.mod(batch_size, multiple), 0)

    return lambda dataset: dataset.filter(_predicate)

def prefetch_element(buffer_size=None):  
    support_auto_tuning = hasattr(tf.data, "experimental") or hasattr(tf.contrib.data, "AUTOTUNE")
    if not support_auto_tuning and buffer_size is None:
        buffer_size = 1
    return lambda dataset: dataset.prefetch(buffer_size)

def build_tf_idf_table(vocab_path, vocab_size, domain_numb, corpora_path):
    print(corpora_path)
    assert domain_numb == len(corpora_path)
    
    words = OrderedDict()    
    vocab_file = open(vocab_path,"r")
    vocab = vocab_file.readlines()
    for l in vocab:
        w = l.strip()
        words[w]=[0]*domain_numb
    for i in range(domain_numb):
        corpora = open(corpora_path[i],"r")
        lines = corpora.readlines()
        line_numb = len(lines)
        for l in lines:
            for w in l.strip().split():
                words[w][i] +=1 
    tf_idf_table = np.zeros((vocab_size, domain_numb))
    keys = list(words.keys())
    corpora_size = np.array([get_dataset_size(path) for path in corpora_path], dtype=np.float32)
    for i in range(vocab_size):
        if i<=2:
            tf_idf_table[i,:] = corpora_size / np.sum(corpora_size)
            continue
        tf_idf_table[i,:] = np.array(words[keys[i]], dtype=np.float32)
        tf_idf_table[i,:] = tf_idf_table[i,:]/np.sum(tf_idf_table[i,:])
    return tf.constant(tf_idf_table, name="tf_idf_table", dtype=tf.float32)

def load_vocab(vocab_path, vocab_size):
    if not vocab_size:
        vocab_size = count_lines(vocab_path) + 1 #for UNK
    vocab = tf.contrib.lookup.index_table_from_file(vocab_path, vocab_size = vocab_size - 1, num_oov_buckets = 1)
    return vocab, vocab_size
   
def load_data(src_path, src_vocab, tag_path=None, batch_size=32, batch_type ="examples", batch_multiplier = 1, tgt_path=None, tgt_vocab=None, 
              max_len=50, bucket_width = 1, mode="Training", padded_shapes = None, 
              shuffle_buffer_size = None, prefetch_buffer_size = 100000, num_threads = 4, version=None, distribution=None, tf_idf_table=None):

    batch_size = batch_size * batch_multiplier
    print("batch_size", batch_size)
    
    def _make_dataset(text_path):
        dataset = tf.data.TextLineDataset(text_path)
        dataset = dataset.map(lambda x: tf.string_split([x]).values) #split by spaces
        return dataset    
       
    def _batch_func(dataset):
        return dataset.padded_batch(batch_size,
                                    padded_shapes=padded_shapes or get_padded_shapes(dataset))

    def _key_func(dataset):                
        #bucket_id = tf.squeeze(dataset["domain"])
        features_length = dataset["src_length"] #features_length_fn(features) if features_length_fn is not None else None
        labels_length = dataset["tgt_length"] #labels_length_fn(labels) if labels_length_fn is not None else None        
        bucket_id = tf.constant(0, dtype=tf.int32)
        if features_length is not None:
            bucket_id = tf.maximum(bucket_id, features_length // bucket_width)
        if labels_length is not None:
            bucket_id = tf.maximum(bucket_id, labels_length // bucket_width)
        return tf.cast(bucket_id, tf.int64)
        #return tf.to_int64(bucket_id)

    def _reduce_func(unused_key, dataset):
        return _batch_func(dataset)

    def _window_size_func(key):
        if bucket_width > 1:
            key += 1  # For bucket_width == 1, key 0 is unassigned.
        size = batch_size // (key * bucket_width)
        if batch_multiplier > 1:
            # Make the window size a multiple of batch_multiplier.
            size = size + batch_multiplier - size % batch_multiplier
        return tf.to_int64(tf.maximum(size, batch_multiplier))             
    
    bos = tf.constant([constants.START_OF_SENTENCE_ID], dtype=tf.int64)
    eos = tf.constant([constants.END_OF_SENTENCE_ID], dtype=tf.int64)
    
    if version==None:
        print("old dataprocessing version")
        src_dataset = _make_dataset(src_path)    
        tag_dataset = _make_dataset(tag_path)
        if mode=="Training":
            tgt_dataset = _make_dataset(tgt_path)
            dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset, tag_dataset))
        elif mode=="Inference":
            dataset = tf.data.Dataset.zip((src_dataset, tag_dataset))
        elif mode == "Predict":
            dataset = src_dataset

        if mode=="Training":                    
            dataset = dataset.map(lambda x,y,tag:{
                    "domain": tf.reshape(tf.random.categorical(tf.expand_dims(tf.log(tf.string_to_number(tag,out_type=tf.float32)),0),1),[]),                
                    "src_raw": x,
                    "tgt_raw": y,
                    "src_ids": src_vocab.lookup(x),
                    "tgt_ids": tgt_vocab.lookup(y),
                    "tgt_ids_in": tf.concat([bos, tgt_vocab.lookup(y)], axis=0),
                    "tgt_ids_out": tf.concat([tgt_vocab.lookup(y), eos], axis=0),
                    "src_length": tf.shape(src_vocab.lookup(x))[0],
                    "tgt_length": tf.shape(tgt_vocab.lookup(y))[0],                
                    }, num_parallel_calls=num_threads)    
                       
        elif mode == "Inference":            
            dataset = dataset.map(lambda x,tag:{
                    "domain": tf.string_to_number(tag[0],out_type=tf.int64),
                    "src_raw": x,                
                    "src_ids": src_vocab.lookup(x),                
                    "src_length": tf.shape(src_vocab.lookup(x))[0],                
                    }, num_parallel_calls=num_threads) 
            
        elif mode == "Predict":            
            dataset = dataset.map(lambda x:{
                    "src_raw": x,                
                    "src_ids": src_vocab.lookup(x),                
                    "src_length": tf.shape(src_vocab.lookup(x))[0],                
                    }, num_parallel_calls=num_threads)
            
        if mode=="Training":            
            if shuffle_buffer_size is not None and shuffle_buffer_size != 0:            
                dataset_size = get_dataset_size(src_path) 
                if dataset_size is not None:
                    if shuffle_buffer_size < 0:
                        shuffle_buffer_size = dataset_size
                elif shuffle_buffer_size < dataset_size:        
                    dataset = dataset.apply(random_shard(shuffle_buffer_size, dataset_size))        
                dataset = dataset.shuffle(shuffle_buffer_size)

            dataset = dataset.filter(lambda x: tf.logical_and(tf.logical_and(tf.greater(x["src_length"],0), tf.greater(x["tgt_length"], 0)), tf.logical_and(tf.less_equal(x["src_length"], max_len), tf.less_equal(x["tgt_length"], max_len))))
            
            if bucket_width is None:
                dataset = dataset.apply(_batch_func)
            else:
                if hasattr(tf.data, "experimental"):
                    group_by_window_fn = tf.data.experimental.group_by_window
                else:
                    group_by_window_fn = tf.contrib.data.group_by_window
                print("batch type: ", batch_type)
                if batch_type == "examples":
                    dataset = dataset.apply(group_by_window_fn(_key_func, _reduce_func, window_size = batch_size))
                elif batch_type == "tokens":
                    dataset = dataset.apply(group_by_window_fn(_key_func, _reduce_func, window_size_func = _window_size_func))   
                else:
                    raise ValueError(
                            "Invalid batch type: '{}'; should be 'examples' or 'tokens'".format(batch_type))
            dataset = dataset.apply(filter_irregular_batches(batch_multiplier))             
            dataset = dataset.repeat()
            dataset = dataset.apply(prefetch_element(buffer_size=prefetch_buffer_size))            
            
        else:
            dataset = dataset.apply(_batch_func)
                        
    elif version==1:
        print("new dataprocessing version")
        if True:
            if mode == "Training":
                src_datasets = [None] * len(src_path)
                tgt_datasets = [None] * len(tgt_path)
                tag_datasets = [None] * len(src_path)
                datasets = [None] * len(src_path)
                for i in range(len(src_path)):
                    src_datasets[i] = _make_dataset(src_path[i])    
                    tgt_datasets[i] = _make_dataset(tgt_path[i])
                    tag_datasets[i] = _make_dataset(tag_path[i])
                    datasets[i] = tf.data.Dataset.zip((src_datasets[i], tgt_datasets[i], tag_datasets[i]))
            elif mode == "Inference":
                src_dataset = _make_dataset(src_path)
                tag_dataset = _make_dataset(tag_path)
                dataset = tf.data.Dataset.zip((src_dataset, tag_dataset))
            elif mode == "Predict":
                src_dataset = _make_dataset(src_path)
                dataset = src_dataset
        
            if mode=="Training":        
                for i in range(len(src_path)):
                    datasets[i] = datasets[i].map(lambda x,y,tag:{
                        "domain": tf.reshape(tf.random.categorical(tf.expand_dims(tf.log(tf.string_to_number(tag,out_type=tf.float32)),0),1),[]),                        
                        "src_raw": x,
                        "tgt_raw": y,
                        "src_ids": src_vocab.lookup(x),
                        "tgt_ids": tgt_vocab.lookup(y),
                        "tgt_ids_in": tf.concat([bos, tgt_vocab.lookup(y)], axis=0),
                        "tgt_ids_out": tf.concat([tgt_vocab.lookup(y), eos], axis=0),
                        "src_length": tf.shape(src_vocab.lookup(x))[0],
                        "tgt_length": tf.shape(tgt_vocab.lookup(y))[0],                
                        }, num_parallel_calls=num_threads)  
              
            elif mode == "Inference":
                dataset = dataset.map(lambda x,tag:{
                        "domain": tf.string_to_number(tag[0],out_type=tf.int64),
                        "src_raw": x,                
                        "src_ids": src_vocab.lookup(x),                
                        "src_length": tf.shape(src_vocab.lookup(x))[0],                
                        }, num_parallel_calls=num_threads) 
            elif mode == "Predict":
                dataset = dataset.map(lambda x:{
                    "src_raw": x,                
                    "src_ids": src_vocab.lookup(x),                
                    "src_length": tf.shape(src_vocab.lookup(x))[0],                
                    }, num_parallel_calls=num_threads)

            if mode=="Training":
                dataset_size = [None] * len(src_path)
                for i in range(len(src_path)):                
                    if shuffle_buffer_size is not None and shuffle_buffer_size != 0:            
                        dataset_size[i] = get_dataset_size(src_path[i]) 
                        if dataset_size[i] is not None:
                            if shuffle_buffer_size < 0:
                                shuffle_buffer_size = dataset_size[i]
                        elif shuffle_buffer_size < dataset_size[i]:        
                            dataset = dataset.apply(random_shard(shuffle_buffer_size, dataset_size[i]))        
                   
                        datasets[i] = datasets[i].shuffle(shuffle_buffer_size)
                        datasets[i] = datasets[i].filter(lambda x: tf.logical_and(tf.logical_and(tf.greater(x["src_length"],0), tf.greater(x["tgt_length"], 0)), tf.logical_and(tf.less_equal(x["src_length"], max_len), tf.less_equal(x["tgt_length"], max_len))))
                        if bucket_width is None:
                            datasets[i] = datasets[i].apply(_batch_func)
                        else:
                            if hasattr(tf.data, "experimental"):
                                group_by_window_fn = tf.data.experimental.group_by_window
                            else:
                                group_by_window_fn = tf.contrib.data.group_by_window
                            print("batch type: ", batch_type)
                            if batch_type == "examples":
                                datasets[i] = datasets[i].apply(group_by_window_fn(_key_func, _reduce_func, window_size = batch_size))
                            elif batch_type == "tokens":
                                datasets[i] = datasets[i].apply(group_by_window_fn(_key_func, _reduce_func, window_size_func = _window_size_func))   
                            else:
                                raise ValueError(
                                         "Invalid batch type: '{}'; should be 'examples' or 'tokens'".format(batch_type))
                            datasets[i] = datasets[i].apply(filter_irregular_batches(batch_multiplier))             
                            datasets[i] = datasets[i].repeat()
                            datasets[i] = datasets[i].apply(prefetch_element(buffer_size=prefetch_buffer_size))
                    
                if distribution == "Natural":
                    total_size = sum(dataset_size)
                    print([float(_size)/total_size for _size in dataset_size])
                    dataset = tf.contrib.data.sample_from_datasets(datasets, weights=[float(_size)/total_size for _size in dataset_size])
                elif distribution == "Balanced":
                    dataset = tf.contrib.data.sample_from_datasets(datasets, weights=[1.0/len(dataset_size) for _size in dataset_size])               
                elif distribution == "Chronicle":
                    choice_dataset = tf.data.Dataset.range(len(dataset_size)).repeat()
                    dataset = tf.contrib.data.choose_from_datasets(datasets, choice_dataset)                                                           
               
            else:
                dataset = dataset.apply(_batch_func)      
        
        
    return dataset.make_initializable_iterator()
    
    
    
    
    
    
    
    



        

