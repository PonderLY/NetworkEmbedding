"""
Python implementation of Crossmap
Training along different edge types
sigmoid_cross_entropy_with_logits
write embedding
add context embedding sm_w_t sm_b
add degree_sampler and weight_sampler
20180810
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
import types
import sys
from evaluator import QuantitativeEvaluator, QualitativeEvaluator
from vose_sampler import VoseAlias


class CrossData(object):
    def __init__(self):
        self.node_dict, self.node_type = self.read_nodes()
        self.node_num = len(self.node_dict)        
        self.line_num, self.linked_nodes, self.non_linked_nodes, self.et2net = self.read_edges(self.node_num)
        self.node_degree = self.get_degree()

    def read_nodes(self):
        """
        Read node_dict file.
        Input format:
            node_id node_type intype_id value
            seperated by '\x01'
        Output format:
            node_id2nt is a dictionary whose key is the node id and
                value is a list [node_type, intype_id, value]
            node_type is a dictionary whose key is the node type and 
                value is a node_id list of this type
        """
        with open(config.node_dict_path, "r") as f:
            node_lines = f.readlines()
        node_id2nt = {}
        node_type = {}
        for t in ['t', 'w', 'l']:
            node_type[t] = []
        for line in node_lines:
            line_lst = line.split('\x01')
            node_id = int(line_lst[0])
            node_id2nt[node_id] = [line_lst[1], line_lst[2], line_lst[3]]
            node_type[line_lst[1]].append(node_id)
        return node_id2nt, node_type

    def read_edges(self, n_node):
        """
        Read edge file.
        Input format:
            edge_type start_node_id end_node_id weight
            seperated by ' '
        Output format:
            line_num is the num of edges
            linked_nodes is a dictionary
                the first key is the node id
                the second key is the end_node_type
                value is a end_node list 
            non_linked_nodes is a dictionary
                the first key is the node id
                the second key is the end_node_type
                value is a non_linked_node list 
            et2net is a dictionary
                the first key is the edge type
                the second key is a 2-tuple (start_node, end_node)
                value is the weight
        """
        with open(config.edge_file_path, "r") as f:
            edge_lines = f.readlines()

        linked_nodes = {}
        for j in range(n_node):
            linked_nodes[j] = {} # Initialization empty list for every node
            linked_nodes[j]['t'] = []
            linked_nodes[j]['w'] = []
            linked_nodes[j]['l'] = []
        line_num = 0
        et2net = defaultdict(lambda : defaultdict(float))                
        for line in edge_lines:
            line_num = line_num + 1
            line_lst = line.split()
            linked_nodes[int(line_lst[1])][line_lst[0][1]].append(int(line_lst[2]))
            et2net[line_lst[0]][(int(line_lst[1]),int(line_lst[2]))] = float(line_lst[3])
        non_linked_nodes = {}
        # for j in range(n_node):
        #     non_linked_nodes[j] = {} 
        #     non_linked_nodes[j]['t'] = [i for i in self.node_type['t'] if i not in linked_nodes[j]['t']]
        #     non_linked_nodes[j]['w'] = [i for i in self.node_type['w'] if i not in linked_nodes[j]['w']]
        #     non_linked_nodes[j]['l'] = [i for i in self.node_type['l'] if i not in linked_nodes[j]['l']]
        
        return line_num, linked_nodes, non_linked_nodes, et2net

    def get_degree(self):
        """
        Calculate the node degree limited to edge type
        Output:
            u_degree is a dictionary
                the first key is the edge type
                the second key is the start node id
                the value is the certain edge type degree(sum of weights)
        """
        u_degree = {}
        for edge_tp in config.edge_type:
            u_degree[edge_tp] = {}
            node_list = self.node_type[edge_tp[0]]
            for u in node_list:
                u_degree[edge_tp][u] = 0.0
                for v in self.linked_nodes[u][edge_tp[1]]:
                    u_degree[edge_tp][u] += self.et2net[edge_tp][(u,v)]
        return u_degree

        
class CrossEdge(CrossData):
    def __init__(self):
        CrossData.__init__(self)
        self.edge_type = config.edge_type
        self.embed_dim = config.embed_dim
        self.batch_size = 1
        self.degree_sampler = self.degree_sampler_initial()
        self.weight_sampler = self.weight_sampler_initial()
        print("Sampling Initialization done!")
        self.build_net()
        self.initialize_network()
    

    def train(self):
        print("Start training for {} epoches!".format(config.max_epochs))
        average_loss = 0.0
        for epoch in tqdm.tqdm(range(config.max_epochs)):
            for t in config.train_type:
                center_list, pos_list, neg_list = self.get_feed_data(t)
                feed_dict_g = {self.center_node: center_list, self.pos_node: pos_list, self.neg_node: neg_list}
                g_loss, _ = self.sess.run([self.loss, self.g_updates], feed_dict=feed_dict_g)
                average_loss += g_loss
                
            if epoch % config.epoch_test == 0:
                ms = self.sess.run(self.summary_op, feed_dict=feed_dict_g)
                self.summary_writer.add_summary(ms, epoch)
                print("Epoch{}: g_loss {}".format(epoch, average_loss/config.epoch_test))
                average_loss = 0 
                self.eval_test_mr(epoch)  
                self.write_embedding(config.write_file_path+str(epoch)+'.emb')                     


    def build_net(self):
        with tf.variable_scope('Generator'):
            init = tf.random_normal_initializer(mean=0.0, stddev=1.0, seed=None, dtype=tf.float32)
            # init = tf.constant_initializer(self.node_emd_init) # To use pretrained embedding
            self.node_embed = tf.get_variable(name="node_embed", shape=(self.node_num, self.embed_dim), 
                                                dtype=tf.float32, initializer=init, trainable=True)
            self.sm_w_t = tf.Variable(tf.zeros([self.node_num, self.embed_dim], dtype=tf.float32), name='sm_w_t')
            self.sm_b = tf.Variable(tf.zeros([self.node_num], dtype=tf.float32), name='sm_b')

        # placeholder
        self.center_node = tf.placeholder(tf.int32, shape=[None], name='center_node')
        self.pos_node = tf.placeholder(tf.int32, shape=[None], name='pos_node')
        self.neg_node = tf.placeholder(tf.int32, shape=[None], name='neg_node')
        
        # look up embeddings
        self.center_embedding= tf.nn.embedding_lookup(self.node_embed, self.center_node)  
        self.pos_w = tf.nn.embedding_lookup(self.sm_w_t, self.pos_node)
        self.pos_b = tf.nn.embedding_lookup(self.sm_b, self.pos_node)        
        self.neg_w = tf.nn.embedding_lookup(self.sm_w_t, self.neg_node)
        self.neg_b = tf.nn.embedding_lookup(self.sm_b, self.neg_node)        

        self.pos_logits = tf.reduce_sum(tf.multiply(self.center_embedding, self.pos_w), axis=1) + self.pos_b
        self.pos_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.pos_logits,
                                                                               labels=tf.ones_like(self.pos_logits)))                     
        self.neg_logits = tf.matmul(self.center_embedding, self.neg_w, transpose_b=True) + self.neg_b
        self.neg_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.neg_logits,
                                                                               labels=tf.zeros_like(self.neg_logits)))
        self.loss = (self.pos_loss + self.neg_loss) / self.batch_size
        tf.summary.scalar('g_loss', self.loss)
        g_opt = tf.train.AdamOptimizer(config.lr_gen)
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
        self.summary_writer = tf.summary.FileWriter(config.model_log, self.sess.graph)
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


    def degree_sampler_initial(self):
        degree_sampler = {}
        for et in self.edge_type:
            u_list = self.node_type[et[0]]
            weight_u = [self.node_degree[et][u_id] for u_id in u_list]
            p_u = weight_u/np.linalg.norm(weight_u, ord=1)
            alias_in = dict(zip(u_list, p_u))
            # print(alias_in.items())
            try:
                degree_sampler[et] = VoseAlias(alias_in)
            except:
                print(et)
                print(weight_u)
                print(p_u)
        return degree_sampler


    def weight_sampler_initial(self):
        weight_sampler = {}
        for et in self.edge_type:
            weight_sampler[et] = {}
            for u in self.node_type[et[0]]:
                v_list = self.linked_nodes[u][et[1]]
                weight_v = [self.et2net[et][(u, v_id)] for v_id in v_list]
                p_v = weight_v/np.linalg.norm(weight_v, ord=1) 
                alias_in = dict(zip(v_list, p_v))
                weight_sampler[et][u] = VoseAlias(alias_in)
        return weight_sampler
        

    def get_feed_data(self, et):
        # sample start node from the type et[0] degree distribution
        u = self.degree_sampler[et].alias_generation()
        while len(self.linked_nodes[u][et[1]])==len(self.node_type[et[1]]):
            u = self.degree_sampler[et].alias_generation()
        # sample end node from the adjacency distribution
        v = self.weight_sampler[et][u].alias_generation()
        # sample neg node from the type et[1] degree distribution
        neg = v
        while neg==v or neg in self.linked_nodes[u][et[1]]:
            neg = self.degree_sampler[et[1]+et[0]].alias_generation()
        return [u], [v], [neg]
  

    def read_embedding(self, n_node, n_embed):
        with open(config.embed_init, "r") as f:
            lines = f.readlines()
        node_embed = np.random.rand(n_node, n_embed)
        for line in lines:
            emd = line.split()
            node_embed[int(float(emd[0])), :] = [float(i) for i in emd[1:]]
        return node_embed


    def write_embedding(self, path):
        a = np.array(range(self.node_num)).reshape(-1, 1)
        R = self.sess.run(self.node_embed)
        node_embed = np.hstack([a, R])
        node_embed_list = node_embed.tolist()
        node_embed_str = ["\t".join([str(x) for x in line]) + "\n" for line in node_embed_list]
        with open(path, "w+") as f:
            # lines = [str(node_num) + "\t" + str(embed_dim) + "\n"] + node_embed_str
            lines = node_embed_str            
            f.writelines(lines)


    def eval_test_mr(self, epoch):
        print("Epoch%d:  embedding" % epoch)
        node_embed = self.sess.run(self.node_embed)
        self.mr_predict(node_embed, epoch)


    def mr_predict(self, node_embed, epoch):
        test_data = pickle.load(open(config.test_data, 'r'))
        predictor = pickle.load(open(config.crossmap, 'r'))
        predictor.read_embedding_tf(config, node_embed)

        start_time = time.time()
        for t in config.predict_type:
            evaluator = QuantitativeEvaluator(predict_type=t)
            evaluator.get_ranks_with_output(test_data, predictor, config.result_pre+str(epoch)+t+'.txt')
            mrr, mr = evaluator.compute_mrr()
            print('Type:{} mr: {}, mrr: {} '.format(evaluator.predict_type, mr, mrr))
       


if __name__ == "__main__":
    model = CrossEdge()
    print("Adam Optimizer, lr 1e-2, random initialized, degree sampler and weight sampler!")
    start_time = time.time()   
    # model.eval_test_mr(0)
    model.train()
    model.write_embedding(config.write_file_path+'final.emb')
    print("Model training done, elapsed time {}s".format(time.time()-start_time))   
