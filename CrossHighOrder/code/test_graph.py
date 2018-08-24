import pdb
import pickle
from paras import load_params
import sys
import time
import itertools
from collections import defaultdict
from crossdata import CrossData
from evaluator import QuantitativeEvaluator, QualitativeEvaluator




if __name__ == "__main__":
    para_file = sys.argv[1]
    pd = load_params(para_file)  # load parameters as a dict
    g = CrossData(pd)

    test_data = pickle.load(open(pd['test_data'], 'r'))
    model = pickle.load(open(pd['crossmap'], 'r'))
    voca = g.wd2id.keys()

    start_time = time.time()
    nt2nodes = {nt:set() for nt in pd['nt_list']}
    et2net = defaultdict(lambda : defaultdict(lambda : defaultdict(float)))	
    for tweet in test_data:
        l = model.lClus.predict([tweet.lat, tweet.lng])
        t = model.tClus.predict([tweet.ts])
        c = tweet.category
        words = [w for w in tweet.words if w in voca] # from text, only retain those words appearing in voca
        nts = pd['nt_list'][1:]
        if 'c' in nts and c not in pd['category_list']:
            nts.remove('c')
        for nt1 in nts:
            nt2nodes[nt1].add(eval(nt1))
            for nt2 in nts:
                if nt1!=nt2:
                    et2net[nt1+nt2][eval(nt1)][eval(nt2)] += 1
        for w in words:
            nt1 = 'w'
            nt2nodes[nt1].add(eval(nt1))
            for nt2 in nts:
                et2net[nt1+nt2][eval(nt1)][eval(nt2)] += 1
                et2net[nt2+nt1][eval(nt2)][eval(nt1)] += 1
        for w1, w2 in itertools.combinations(words, r=2):
            if w1!=w2:
                et2net['ww'][w1][w2] += 1
                et2net['ww'][w2][w1] += 1

    gf = open(pd['test_edges'], 'w')
    edge_num = 0
    for key_et in et2net.keys():
        for key_s in et2net[key_et].keys():
            if key_et[0]=='w':
                keys_id = g.wd2id[key_s]
            else:
                keys_id = g.node_id2id[key_et[0]][key_s]
            for key_t in et2net[key_et][key_s].keys():
                if key_et[1] == 'w':
                    keyt_id = g.wd2id[key_t]
                else:
                    keyt_id = g.node_id2id[key_et[1]][key_t]
                line = [key_et, str(keys_id), str(keyt_id), str(et2net[key_et][key_s][key_t]),'\n']
                edge_num = edge_num + 1
                gf.write(' '.join(line))
    gf.close()
    print 'There are ', edge_num, 'edges in total!'
    