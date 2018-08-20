"""
CrossLine
train_one_epoch
20180815
Author: Liu Yang
"""
import numpy as np 
import tensorflow as tf 
import time
import config
from copy import deepcopy
from collections import defaultdict
import pickle
import random
import tqdm
import math
import pdb
from crossdata import CrossData
from evaluator import QuantitativeEvaluator, QualitativeEvaluator


        
class CrossLine(object):
    def __init__(self, graph):
        self.cur_epoch = 0
        self.g = graph
        self.node_num = graph.node_num
        self.embed_dim = config.embed_dim
        self.batch_size = config.batch_size
        self.learning_rate = config.lr_gen
        self.negative_ratio = config.neg_num
        self.gen_sampling_table()
        print("Sampling Initialization done!")
        self.build_net()
        self.initialize_network()
    

    def train_one_epoch(self, et):
        average_loss = 0.0
        batches = self.batch_iter(et)
        for batch in batches:
            center, context, sign = batch
            feed_dict_g = {self.center_node: center, self.context_node: context, self.sign: sign}
            g_loss, _ = self.sess.run([self.loss, self.g_updates], feed_dict=feed_dict_g)
            average_loss += g_loss         
        print('epoch:{} sum of loss:{!s}'.format(self.cur_epoch, average_loss))
        self.cur_epoch += 1      


    def build_net(self):
        cur_seed = random.getrandbits(32)
        init = tf.contrib.layers.xavier_initializer(uniform=False, seed=cur_seed)        
        with tf.variable_scope('Generator', reuse=None, initializer=init):
            self.node_embed = tf.get_variable(name="node_embed", shape=(self.node_num, self.embed_dim), 
                                                dtype=tf.float32, initializer=init, trainable=True)
            self.context_embed = tf.get_variable(name="context_embed", shape=(self.node_num, self.embed_dim), 
                                                dtype=tf.float32, initializer=init, trainable=True)
            # placeholder
            self.center_node = tf.placeholder(tf.int32, shape=[None], name='center_node')
            self.context_node = tf.placeholder(tf.int32, shape=[None], name='context_node')
            self.sign = tf.placeholder(tf.float32, shape=[None], name='sign')
            
            # look up embeddings
            self.center_embedding= tf.nn.embedding_lookup(self.node_embed, self.center_node)  
            self.context_embedding= tf.nn.embedding_lookup(self.context_embed, self.context_node)  
                   
            self.loss = -tf.reduce_mean(tf.log_sigmoid(self.sign*tf.reduce_sum(tf.multiply(self.center_embedding, self.context_embedding), axis=1)))
            tf.summary.scalar('g_loss', self.loss)
            g_opt = tf.train.AdamOptimizer(self.learning_rate)
            # g_opt = tf.train.GradientDescentOptimizer(config.lr_gen)
            self.g_updates = g_opt.minimize(self.loss)
        

    def initialize_network(self):
        print("Initializing network...")
        # Settings for GPU
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=self.config)

        # self.sess = tf.Session()
        self.summary_op = tf.summary.merge_all()
        # self.summary_writer = tf.summary.FileWriter(config.model_log, self.sess.graph)
        # add the saver
        self.saver = tf.train.Saver()
        # Initialize variables
        self.init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(self.init_op)
        # Restore the model
        # Check if there exists checkpoint, if true, load it
        ckpt = tf.train.get_checkpoint_state(config.model_log)
        if ckpt and ckpt.model_checkpoint_path and config.load_model:
            print("Load the checkpoint: %s" % ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("Model restored...")


    def batch_iter(self, et):
        table_size = 1e8
        edges = self.g.edges[et]
        data_size = len(edges)
        shuffle_indices = np.random.permutation(np.arange(data_size))

        # positive or negative mod
        mod = 0
        mod_size = 1 + self.negative_ratio
        h = []
        t = []
        sign = 0

        start_index = 0
        end_index = min(start_index+self.batch_size, data_size)
        while start_index < data_size:
            if mod == 0:
                sign = 1.
                h = []
                t = []
                for i in range(start_index, end_index):
                    if not random.random() < self.edge_prob[et][shuffle_indices[i]]:
                        shuffle_indices[i] = self.edge_alias[et][shuffle_indices[i]]
                    cur_h = edges[shuffle_indices[i]][0]
                    cur_t = edges[shuffle_indices[i]][1]
                    h.append(cur_h)
                    t.append(cur_t)
            else:
                sign = -1.
                t = []
                for i in range(len(h)):
                    intype_id = self.sampling_table[et][random.randint(0, table_size-1)]
                    sample_id = self.g.node_id2id[et[1]][intype_id]
                    t.append(sample_id)

            yield h, t, [sign]
            mod += 1
            mod %= mod_size
            if mod == 0:
                start_index = end_index
                end_index = min(start_index+self.batch_size, data_size)

    def gen_sampling_table(self):
        table_size = 1e8
        power = 0.75
        print("Pre-procesing for non-uniform negative sampling!")
        self.sampling_table = {}
        self.edge_alias = {}
        self.edge_prob = {}

        for et in config.edge_type:
            numNodes = len(self.g.node_type[et[1]])

            node_degree = np.zeros(numNodes) # out degree

            for edge in self.g.edges[et]:
                node_degree[int(self.g.node_dict[edge[1]][1])] += self.g.et2net[et][edge]

            norm = sum([math.pow(node_degree[i], power) for i in range(numNodes)])

            self.sampling_table[et] = np.zeros(int(table_size), dtype=np.uint32)

            p = 0
            i = 0
            for j in range(numNodes):
                p += float(math.pow(node_degree[j], power)) / norm
                while i < table_size and float(i) / table_size < p:
                    self.sampling_table[et][i] = j
                    i += 1

            data_size = len(self.g.edges[et])
            self.edge_alias[et] = np.zeros(data_size, dtype=np.int32)
            self.edge_prob[et] = np.zeros(data_size, dtype=np.float32)
            large_block = np.zeros(data_size, dtype=np.int32)
            small_block = np.zeros(data_size, dtype=np.int32)
            
            total_sum = sum([self.g.et2net[et][edge] for edge in self.g.edges[et]])     
            norm_prob = [self.g.et2net[et][edge]*data_size/total_sum for edge in self.g.edges[et]]
            num_small_block = 0
            num_large_block = 0
            cur_small_block = 0
            cur_large_block = 0
            for k in range(data_size-1, -1, -1):
                if norm_prob[k] < 1:
                    small_block[num_small_block] = k
                    num_small_block += 1
                else:
                    large_block[num_large_block] = k
                    num_large_block += 1
            while num_small_block and num_large_block:
                num_small_block -= 1
                cur_small_block = small_block[num_small_block]
                num_large_block -= 1
                cur_large_block = large_block[num_large_block]
                self.edge_prob[et][cur_small_block] = norm_prob[cur_small_block]
                self.edge_alias[et][cur_small_block] = cur_large_block
                norm_prob[cur_large_block] = norm_prob[cur_large_block] + norm_prob[cur_small_block] -1
                if norm_prob[cur_large_block] < 1:
                    small_block[num_small_block] = cur_large_block
                    num_small_block += 1
                else:
                    large_block[num_large_block] = cur_large_block
                    num_large_block += 1

            while num_large_block:
                num_large_block -= 1
                self.edge_prob[et][large_block[num_large_block]] = 1
            while num_small_block:
                num_small_block -= 1
                self.edge_prob[et][small_block[num_small_block]] = 1

    def get_embeddings(self):
        node_embeddings = self.node_embed.eval(session=self.sess)
        context_embeddings = self.context_embed.eval(session=self.sess)        
        return node_embeddings, context_embeddings

    def save_model(self, step=None):
        self.saver.save(self.sess, config.model_log, global_step=step)


class CrossEdge(object):
    def __init__(self, graph):
        self.model = CrossLine(graph)
    

    def train_eval(self):
        print("Start training for {} epoches!".format(config.max_epochs))
        for epoch in tqdm.tqdm(range(config.max_epochs)):
            for et in config.train_type:
                self.model.train_one_epoch(et)
                
            if epoch % config.epoch_test == 0:
                self.eval_test_mr(epoch)  
                self.write_embedding(config.write_file_path+str(epoch)+'.emb')    
        
        self.model.save_model()                  
  

    def read_embedding(self, n_node, n_embed):
        with open(config.embed_init, "r") as f:
            lines = f.readlines()
        node_embed = np.random.rand(n_node, n_embed)
        for line in lines:
            emd = line.split()
            node_embed[int(float(emd[0])), :] = [float(i) for i in emd[1:]]
        return node_embed


    def write_embedding(self, path):
        a = np.array(range(self.model.node_num)).reshape(-1, 1)
        R, RR = self.model.get_embeddings()
        node_embed = np.hstack([a, R])
        node_embed_list = node_embed.tolist()
        node_embed_str = ["\t".join([str(x) for x in line]) + "\n" for line in node_embed_list]
        with open(path, "w+") as f:
            # lines = [str(node_num) + "\t" + str(embed_dim) + "\n"] + node_embed_str
            lines = node_embed_str            
            f.writelines(lines)


    def eval_test_mr(self, epoch):
        print("Epoch%d:  embedding" % epoch)
        node_embed, context_embed = self.model.get_embeddings()
        self.mr_predict(node_embed, context_embed, epoch)


    def mr_predict(self, node_embed, context_embed, epoch):
        test_data = pickle.load(open(config.test_data, 'r'))
        predictor = pickle.load(open(config.crossmap, 'r'))
        predictor.read_embedding_tf(config, node_embed, context_embed)

        start_time = time.time()
        for t in config.predict_type:
            evaluator = QuantitativeEvaluator(predict_type=t)
            evaluator.get_ranks_with_output(test_data, predictor, config.result_pre+str(epoch)+t+'.txt')
            mrr, mr = evaluator.compute_mrr()
            print('Type:{} mr: {}, mrr: {} '.format(evaluator.predict_type, mr, mrr))
       


if __name__ == "__main__":
    g = CrossData()
    model = CrossEdge(g)
    print("Adam Optimizer, lr 1e-2, xavier initializer, neg sample 10!")
    start_time = time.time()
    model.train_eval()
    model.write_embedding(config.write_file_path+'final.emb')
    print("Model training done, elapsed time {}s".format(time.time()-start_time))   
