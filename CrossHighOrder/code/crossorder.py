"""
Using high order information to train crossmap C++
20180819
Author: Liu Yang
"""
import numpy as np  
import time
import ast
from copy import deepcopy
from collections import defaultdict
import itertools
import pickle
import random
import math
import os, sys
import pdb
from paras import load_params
from sklearn.preprocessing import normalize
from crossdata import CrossData
from evaluator import QuantitativeEvaluator, QualitativeEvaluator
from subprocess import call, check_call


        
class HighOrder(object):
    def __init__(self, graph, pd):
        self.g = graph
        self.pd = pd
        self.nt2nodes = self.construct_nt2nodes()
        self.et2net = self.construct_et2net()
        self.nt2vecs = None # center vectors
        self.nt2cvecs = None # context vectors


    def construct_nt2nodes(self):
        nt2nodes = {nt:set() for nt in self.pd['nt_list']}
        for n_id in self.g.node_type['t']:
            nt2nodes['t'].add(int(self.g.node_dict[n_id][1]))
        for n_id in self.g.node_type['l']:
            nt2nodes['l'].add(int(self.g.node_dict[n_id][1]))
        for n_id in self.g.node_type['w']:
            nt2nodes['w'].add(self.g.node_dict[n_id][2])
        return nt2nodes


    def construct_et2net(self):
        node_dict = self.g.node_dict
        et2net = defaultdict(lambda : defaultdict(lambda : defaultdict(float)))
        for key_et in self.g.et2net.keys():
            for key_s, key_t in self.g.et2net[key_et].keys():
                if key_et[0]=='w':
                    s = node_dict[key_s][2]
                else:
                    s = int(node_dict[key_s][1])
                if key_et[1]=='w':
                    t = node_dict[key_t][2]
                else:
                    t = int(node_dict[key_t][1])
                et2net[key_et][s][t] = self.g.et2net[key_et][(key_s,key_t)]
        return et2net

        
    def construct_adjacency_matrix(self, et):
        node_dict = self.g.node_dict        
        start_num = len(self.g.node_type[et[0]])
        end_num = len(self.g.node_type[et[1]])        
        adj_matrix = np.zeros(shape=(start_num, end_num), dtype=np.float32)

        for key_s,key_t in self.g.et2net[et].keys():
            s = int(node_dict[key_s][1])
            t = int(node_dict[key_t][1])            
            adj_matrix[s, t] = self.g.et2net[et][(s, t)]

        return normalize(adj_matrix, norm='l1')                
  

    def modify_et2net(self, et, W):
        row, col = W.shape
        for s in xrange(row):
            for t in xrange(col):
                if W[s,t] >1e-2:
                    if et[0]=='w':
                        key_id = self.g.node_id2id['w'][s]
                        key_s = self.g.node_dict[key_id][2]
                    else:
                        key_s = s
                    if et[1]=='w':
                        key_id = self.g.node_id2id['w'][t]
                        key_t = self.g.node_dict[key_id][2]
                    else:
                        key_t = t
                    self.et2net[et][key_s][key_t] = W[s,t]

    def fit(self):
        self.embed_algo = GraphEmbed(self.pd)
        self.nt2vecs, self.nt2cvecs = self.embed_algo.fit(self.nt2nodes, self.et2net, 1000000)


    def mr_predict(self):
        test_data = pickle.load(open(self.pd['test_data'], 'r'))
        predictor = pickle.load(open(self.pd['crossmap'], 'r'))
        predictor.update_vec_cvec(self.nt2vecs, self.nt2cvecs)

        start_time = time.time()
        for t in self.pd['predict_type']:
            evaluator = QuantitativeEvaluator(predict_type=t)
            evaluator.get_ranks(test_data, predictor)
            # evaluator.get_ranks_with_output(test_data, predictor, config.result_pre+str(epoch)+t+'.txt')
            mrr, mr = evaluator.compute_mrr()
            print('Type:{} mr: {}, mrr: {} '.format(evaluator.predict_type, mr, mrr))


class GraphEmbed(object):
	def __init__(self, pd):
		self.pd = pd
		self.nt2vecs = dict()
		self.nt2cvecs = dict()
		self.path_prefix = 'GraphEmbed/'
		self.path_suffix = '-'+str(os.getpid())+'.txt'

	def fit(self, nt2nodes, et2net, sample_size):
		self.write_line_input(nt2nodes, et2net)
		self.execute_line(sample_size)
		self.read_line_output()
		return self.nt2vecs, self.nt2cvecs

	def write_line_input(self, nt2nodes, et2net):
		if 'c' not in nt2nodes: # add 'c' nodes (with no connected edges) to comply to Line's interface
			nt2nodes['c'] = self.pd['category_list']
		for nt, nodes in nt2nodes.items():
			# print nt, len(nodes)
			node_file = open(self.path_prefix+'node-'+nt+self.path_suffix, 'w')
			for node in nodes:
				node_file.write(str(node)+'\n')
		all_et = [nt1+nt2 for nt1, nt2 in itertools.product(nt2nodes.keys(), repeat=2)]
		for et in all_et:
			edge_file = open(self.path_prefix+'edge-'+et+self.path_suffix, 'w')
			if et in et2net:
				for u, u_nb in et2net[et].items():
					for v, weight in u_nb.items():
						edge_file.write('\t'.join([str(u), str(v), str(weight), 'e'])+'\n')

	def execute_line(self, sample_size):
		command = ['./hin2vec']
		command += ['-size', str(self.pd['dim'])]
		command += ['-negative', str(self.pd['negative'])]
		command += ['-alpha', str(self.pd['alpha'])]
		sample_num_in_million = max(1, sample_size/1000000)
		command += ['-samples', str(sample_num_in_million)]
		command += ['-threads', str(10)]
		command += ['-second_order', str(self.pd['second_order'])]
		command += ['-job_id', str(os.getpid())]
		# call(command, cwd=self.path_prefix, stdout=open('stdout.txt','wb'))
		call(command, cwd=self.path_prefix)

	def read_line_output(self):
		for nt in self.pd['nt_list']:
			for nt2vecs,vec_type in [(self.nt2vecs,'output-'), (self.nt2cvecs,'context-')]:
				vecs_path = self.path_prefix+vec_type+nt+self.path_suffix
				vecs_file = open(vecs_path, 'r')
				vecs = dict()
				for line in vecs_file:
					node, vec_str = line.strip().split('\t')
					try:
						node = ast.literal_eval(node)
					except: # when nt is 'w', the type of node is string
						pass
					vecs[node] = np.array([float(i) for i in vec_str.split(' ')])
				nt2vecs[nt] = vecs
		for f in os.listdir(self.path_prefix): # clean up the tmp files created by this execution
		    if f.endswith(self.path_suffix):
		        os.remove(self.path_prefix+f)
       


if __name__ == "__main__":
    para_file = sys.argv[1]
    pd = load_params(para_file)  # load parameters as a dict
    g = CrossData(pd)
    model = HighOrder(g, pd)
    W = model.construct_adjacency_matrix('ww')
    model.modify_et2net('ww', W*W)
    print("Start training!")
    start_time = time.time()
    model.fit()
    print("Model training done, elapsed time {}s".format(time.time()-start_time))   
    model.mr_predict()