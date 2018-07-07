"""
Python implementation of Crossmap
Training along different edge types
No tensorflow
Explicit update formulation
20180706
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
import scipy
from sklearn.preprocessing import normalize
from evaluator import QuantitativeEvaluator, QualitativeEvaluator



class CrossNet(object):
    def __init__(self):
        self.node_dict, self.node_type = self.read_nodes()
        self.node_num = len(self.node_dict)
        self.embed_dim = config.embed_dim
        self.batch_size = config.batch_size
        self.line_num, self.linked_nodes, self.non_linked_nodes, self.et2net = self.read_edges(self.node_num)
        self.node_emd = np.random.rand(self.node_num, self.embed_dim)
        # self.node_emd_init = np.zeros((self.node_num, self.embed_dim), dtype=np.float32)        
        # self.node_emd_init = self.read_embedding(self.node_num, self.embed_dim)
    

    def train(self):
        print("Start training for {} epoches!".format(config.max_epochs))
        for epoch in tqdm.tqdm(range(config.max_epochs)):
            for t in config.train_type:
                t_edge = self.et2net[t].keys()
                for u,v in t_edge:
                    self.update(t, u, v)
                
            if epoch % config.epoch_test == 0:
                # print("Epoch{}: g_loss {}".format(epoch, g_loss)) 
                self.eval_test_mr(epoch)                       


    def update(self, et, u, v):
        alpha = config.lr_gen
        MAX_EXP = 6
        error_vec = np.zeros(config.embed_dim)
        for d in range(config.neg_num+1):
            if d == 0:
                target = v
                label = 1
            else:
                target = self.neg_sampling(u, v, et)
                label = 0
            f = np.vdot(self.node_emd[u,:], self.node_emd[target,:])
            if f > MAX_EXP:
                g = (label - 1) * alpha
            elif f < -MAX_EXP:
                g = (label - 0) * alpha
            else:
                g = (label - scipy.special.expit(f)) * alpha
            error_vec += g * self.node_emd[target,:]
            self.node_emd[v, :] += g * self.node_emd[u, :]
        self.node_emd[u,:] += error_vec


    def neg_sampling(self, u, v, et):
        v_nb = self.linked_nodes[v][et[0]]
        weight_v = [self.et2net[et[1]+et[0]][(v, v_)] for v_ in v_nb]
        sample_prob = normalize([weight_v], norm='l1')[0]
        return np.random.choice(v_nb, size=1, p=sample_prob)[0]


    def get_feed_data(self, edge_list, et):
        center_list = []
        pos_list = []
        neg_list = []
        # s is a 2-tuple
        for s in edge_list:
            if len(self.non_linked_nodes[s[0]][et[1]])<1: # there is no neg for node 
                continue
            else:
                center_list.append(s[0])
                pos_list.append(s[1])
                neg_sample = random.sample(self.non_linked_nodes[s[0]][et[1]], config.neg_num) # only 1 negtive sample
                neg_list.append(neg_sample[0])
        return center_list, pos_list, neg_list

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


    def read_nodes(self):
        """
        Read node_dict file.
        Input format:
            node_id node_type intype_id value
            seperated by '\x01'
        Output format:
            a dictionary
            key is the node id
            value is a list [node_type, intype_id, value]
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


    def read_embedding(self, n_node, n_embed):
        with open(config.embed_init, "r") as f:
            lines = f.readlines()
        node_embed = np.random.rand(n_node, n_embed)
        for line in lines:
            emd = line.split()
            node_embed[int(float(emd[0])), :] = [float(i) for i in emd[1:]]

        return node_embed


    def try1(self):
        feed_dict_g = {self.center_node: [10], self.pos_node: [4, 6], self.neg_node: [2, 3], self.pos_num: [2]}
        c_pos, c_neg = self.sess.run([self.c_pos_product, self.c_neg_product], feed_dict=feed_dict_g)
        print(c_pos, c_neg)



    def eval_test_mr(self, epoch):
        print("Epoch%d:  embedding" % epoch)
        # node_embed = self.sess.run(self.node_embed)
        self.mr_predict(self.node_emd)


    def mr_predict(self, node_embed):
        test_data = pickle.load(open(config.test_data, 'r'))
        predictor = pickle.load(open(config.crossmap, 'r'))
        predictor.read_embedding_tf(config, node_embed)

        start_time = time.time()
        for t in config.predict_type:
            evaluator = QuantitativeEvaluator(predict_type=t)
            evaluator.get_ranks(test_data, predictor)
            mrr, mr = evaluator.compute_mrr()
            print('Type:{} mr: {}, mrr: {} '.format(evaluator.predict_type, mr, mrr))
       


if __name__ == "__main__":
    model = CrossNet()
    print("Explicit update, lr 1e-3, random initialized, log sigmoid loss!")
    start_time = time.time()   
    # model.eval_test_mr(0)
    model.train()
    print("Model training done, elapsed time {}s".format(time.time()-start_time))   
