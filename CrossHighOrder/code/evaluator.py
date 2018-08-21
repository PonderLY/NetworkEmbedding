import bisect
import pdb
import urllib2, time
from embed import *

class QuantitativeEvaluator:
    def __init__(self, predict_type='w', real_num=3, fake_num=7):
        self.ranks = []
        self.predict_type = predict_type
        self.fake_num = fake_num
        if self.predict_type=='p':
            self.pois = io.read_pois()

    def get_ranks(self, tweets, predictor, graph):
        noiseList = np.random.choice((self.pois if self.predict_type=='p' else tweets), self.fake_num*len(tweets)).tolist()
        for tweet in tweets:
            t_id, l_id, w_id = self.get_global_id(tweet, predictor, graph)
            if self.predict_type=='t':
                l_set = set(graph.linked_nodes[l_id]['t'])
                w_len = [len(graph.linked_nodes[w]['t']) for w in w_id]
                w_ind = w_len.index(max(w_len))
                w_set = set(graph.linked_nodes[w_id[w_ind]]['t'])
                t_candi = l_set & w_set
                print('t', len(t_candi), '\n')
            elif self.predict_type=='l':
                t_set = set(graph.linked_nodes[t_id]['l'])
                w_len = [len(graph.linked_nodes[w]['l']) for w in w_id]
                w_ind = w_len.index(max(w_len))
                w_set = set(graph.linked_nodes[w_id[w_ind]]['l'])
                l_candi = t_set & w_set
                print('l', len(l_candi), '\n')
            elif self.predict_type=='w':
                t_set = set(graph.linked_nodes[t_id]['w'])                
                l_set = set(graph.linked_nodes[l_id]['w'])
                w_candi = t_set & l_set
                print('w', len(w_candi), '\n')

            scores = []
            if self.predict_type=='p':
                score = predictor.predict(tweet.ts, tweet.poi_lat, tweet.poi_lng, tweet.words, tweet.category, self.predict_type)
            else:
                score = predictor.predict(tweet.ts, tweet.lat, tweet.lng, tweet.words, tweet.category, self.predict_type)
            scores.append(score)
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