import bisect
import copy
import pdb
import urllib2, time
from embed import *

class QuantitativeEvaluator:
    def __init__(self, predict_type='w', real_num=3, fake_num=7):
        self.ranks = []
        self.highest_ranks = []
        self.predict_type = predict_type
        self.real_num = real_num
        self.fake_num = fake_num
        if self.predict_type=='p':
            self.pois = io.read_pois()

    def get_ranks(self, tweets, predictor, graph):
        noiseList = np.random.choice((self.pois if self.predict_type=='p' else tweets), self.fake_num*len(tweets)).tolist()
        link_dict = graph.linked_nodes  
        # link_dict = graph.linked_nodes   
        test_num = 0   
        for tweet in tweets:
            t_id, l_id, w_id = self.get_global_id(tweet, predictor, graph)
            tweet_id = {'t':t_id, 'l':l_id, 'w':w_id}
            if self.predict_type=='t':
                t_candi = set(link_dict[l_id]['t'])
                for w in w_id:
                    w_set = set(link_dict[w]['t'])
                    t_candi = t_candi & w_set
                if len(t_candi) > 15:
                    continue
                else:
                    ans = self.get_top_k_candi(t_candi, self.predict_type, tweet_id, graph)
                # print('t', len(t_candi), '\n')
            elif self.predict_type=='l':
                l_candi = set(link_dict[t_id]['l'])
                for w in w_id:
                    w_set = set(link_dict[w]['l'])
                    l_candi = l_candi & w_set
                if len(l_candi) > 50:
                    continue
                else:
                    ans = self.get_top_k_candi(l_candi, self.predict_type, tweet_id, graph)
                # print('l', len(l_candi), '\n')
            elif self.predict_type=='w':
                t_set = set(link_dict[t_id]['w'])                
                l_set = set(link_dict[l_id]['w'])
                w_candi = t_set & l_set
                if len(w_candi) > 500:
                    continue
                else:
                    ans = self.get_top_k_candi(w_candi, self.predict_type, tweet_id, graph)
                # print('w', len(w_candi), '\n')
            test_num += 1

            scores = []
            score = predictor.predict(tweet.ts, tweet.lat, tweet.lng, tweet.words, tweet.category, self.predict_type)
            scores.append(score)
            true_scores = []
            if ans is not None:
                for k_id in ans:
                    if self.predict_type=='t':
                        ts = float(graph.node_dict[k_id][2].strip('[]'))
                        true_scores.append(predictor.predict(ts, tweet.lat, tweet.lng, tweet.words, tweet.category, self.predict_type))
                    elif self.predict_type=='l':
                        loc_str = graph.node_dict[k_id][2]
                        loc = [float(x) for x in loc_str.strip('[]').split()]
                        true_scores.append(predictor.predict(tweet.ts, loc[0], loc[1], tweet.words, tweet.category, self.predict_type))
                    elif self.predict_type=='w':
                        word = graph.node_dict[k_id][2]
                        true_scores.append(predictor.predict(tweet.ts, tweet.lat, tweet.lng, [word], tweet.category, self.predict_type))
            scores.extend(true_scores)
            for i in range(self.fake_num):
                noise = noiseList.pop()
                if self.predict_type in ['l','p']:
                    noise_score = predictor.predict(tweet.ts, noise.lat, noise.lng, tweet.words, tweet.category, self.predict_type)
                elif self.predict_type=='t':
                    noise_score = predictor.predict(noise.ts, tweet.lat, tweet.lng, tweet.words, tweet.category, self.predict_type)
                elif self.predict_type=='w':
                    noise_score = predictor.predict(tweet.ts, tweet.lat, tweet.lng, noise.words, tweet.category, self.predict_type)
                scores.append(noise_score)
            scores.sort()
            # handle ties
            rank = len(scores)+1-(bisect.bisect_left(scores,score)+bisect.bisect_right(scores,score)+1)/2.0
            self.ranks.append(rank)
            true_scores.append(score)
            score = max(true_scores)
            highest_rank = len(scores)+1-(bisect.bisect_left(scores,score)+bisect.bisect_right(scores,score)+1)/2.0
            self.highest_ranks.append(highest_rank)
        print('There are {} tweets for test, rank computation done!'.format(test_num))

    def get_ranks_with_output(self, tweets, predictor, pd):
        noiseList = np.random.choice((self.pois if self.predict_type=='p' else tweets), self.fake_num*len(tweets)).tolist()
        evaluate_f = open(pd['evaluate_file'], 'w')
        for tweet in tweets:
            scores = []
            if self.predict_type=='p':
                score = predictor.predict(tweet.ts, tweet.poi_lat, tweet.poi_lng, tweet.words, tweet.category, self.predict_type)
            else:
                score = predictor.predict(tweet.ts, tweet.lat, tweet.lng, tweet.words, tweet.category, self.predict_type)
            scores.append(score)
            t_line = [tweet.datetime, tweet.ts, tweet.lat, tweet.lng, tweet.words, tweet.category, self.predict_type]
            t_line_str = "\t".join([str(x) for x in t_line]) + "\n" + "Its score is {}.\t The followings are noises.\n".format(score)
            evaluate_f.write(t_line_str)

            for i in range(self.fake_num):
                noise = noiseList.pop()
                if self.predict_type in ['l','p']:
                    noise_score = predictor.predict(tweet.ts, noise.lat, noise.lng, tweet.words, tweet.category, self.predict_type)
                elif self.predict_type=='t':
                    noise_score = predictor.predict(noise.ts, tweet.lat, tweet.lng, tweet.words, tweet.category, self.predict_type)
                elif self.predict_type=='w':
                    noise_score = predictor.predict(tweet.ts, tweet.lat, tweet.lng, noise.words, tweet.category, self.predict_type)
                scores.append(noise_score)
                s_line = [noise.datetime, noise.ts, noise.lat, noise.lng, noise.words]
                s_line_str = "noise {} \n".format(i) + "\t".join([str(x) for x in s_line]) + "\n" + "Its score is {}.\n".format(noise_score)
                evaluate_f.write(s_line_str)

            scores.sort()
            # handle ties
            rank = len(scores)+1-(bisect.bisect_left(scores,score)+bisect.bisect_right(scores,score)+1)/2.0
            self.ranks.append(rank)
            evaluate_f.write("Its rank is {}.\n\n".format(rank))

        evaluate_f.close()

    def compute_mrr(self):
        ranks = self.ranks
        rranks = [1.0/rank for rank in ranks]
        mrr,mr = sum(rranks)/len(rranks),sum(ranks)/len(ranks)
        return round(mrr,4), round(mr,4)

    def compute_highest_mrr(self):
        ranks = self.highest_ranks
        rranks = [1.0/rank for rank in ranks]
        mrr,mr = sum(rranks)/len(rranks),sum(ranks)/len(ranks)
        return round(mrr,4), round(mr,4)

    def get_global_id(self, tweet, predictor, graph):
        t_local = predictor.tClus.predict([tweet.ts])
        t_id = graph.node_id2id['t'][t_local]
        l_local = predictor.lClus.predict([tweet.lat, tweet.lng])
        l_id = graph.node_id2id['l'][l_local]
        w_id = []
        voca = graph.wd2id.keys()
        for w in tweet.words:
            if w in voca:
                w_id.append(graph.wd2id[w])
        return t_id, l_id, w_id      

    def get_top_k_candi(self, candi_id, predict_type, tweet_id, graph):
        candi = list(candi_id)
        if len(candi_id)<1:
            return None
        elif len(candi_id)<=self.real_num:
            return candi
        else:
            sumofwei = []
            if predict_type=='t':
                for t_id in candi:
                    t_wei = graph.et2net['tl'][(t_id, tweet_id['l'])]
                    for w_id in tweet_id['w']:
                        t_wei += graph.et2net['tw'][(t_id, w_id)]
                    sumofwei.append(t_wei)
            elif predict_type=='l':
                for l_id in candi:
                    l_wei = graph.et2net['lt'][(l_id, tweet_id['t'])]
                    for w_id in tweet_id['w']:
                        l_wei += graph.et2net['lw'][(l_id, w_id)]
                    sumofwei.append(l_wei)
            else:
                for w_id in candi:
                    w_wei = graph.et2net['wt'][(w_id, tweet_id['t'])]
                    w_wei += graph.et2net['wl'][(w_id, tweet_id['l'])]
                    sumofwei.append(w_wei)
            wei_copy = copy.deepcopy(sumofwei)
            wei_copy.sort(reverse=True)
            ans_index = [sumofwei.index(k) for k in wei_copy[:self.real_num]]
            return [candi[k] for k in ans_index]
            



class QualitativeEvaluator:
    def __init__(self, predictor, output_dir):
        self.predictor = predictor
        self.output_dir = output_dir

    def plot_locations_on_google_map(self, locations, output_path):
        request ='https://maps.googleapis.com/maps/api/staticmap?zoom=10&size=600x600&maptype=roadmap&'
        for lat, lng in locations:
            request += 'markers=color:red%7C' + '%f,%f&' % (lat, lng)
        proxy = urllib2.ProxyHandler({})
        opener = urllib2.build_opener(proxy)
        response = opener.open(request).read()
        with open(output_path, 'wb') as f:
            f.write(response)
            f.close()
        time.sleep(3)

    def scribe(self, directory, ls, ts, ws, show_ls):
        if not os.path.isdir(directory):
            os.makedirs(directory)
        for nbs, file_name in [(ts, 'times.txt'), (ws, 'words.txt')]:
            output_file = open(directory+file_name,'w')
            for nb in nbs:
                output_file.write(str(nb)+'\n')
        if show_ls:
            self.plot_locations_on_google_map(ls[:10], directory+'locations.png')
        else:
            self.plot_locations_on_google_map(ls[:1], directory+'queried_location.png')

    def getNbs1(self, query):
        if type(query)==str and query.lower() not in self.predictor.nt2vecs['w']:
            print query, 'not in voca'
            return
        directory = self.output_dir+str(query)+'/'
        ls, ts, ws = [self.predictor.get_nbs1(query, nt) for nt in ['l', 't', 'w']]
        self.scribe(directory, ls, ts, ws, type(query)!=list)

    def getNbs2(self, query1, query2, func=lambda a, b:a+b):
        if type(query1)==str and query1.lower() not in self.predictor.nt2vecs['w']:
            print query1, 'not in voca'
            return
        if type(query2)==str and query2.lower() not in self.predictor.nt2vecs['w']:
            print query2, 'not in voca'
            return
        directory = self.output_dir+str(query1)+'-'+str(query2)+'/'
        ls, ts, ws = [self.predictor.get_nbs2(query1, query2, func, nt) for nt in ['l', 't', 'w']]
        self.scribe(directory, ls, ts, ws, type(query1)!=list)