from django.shortcuts import render
from django.template.loader import get_template
from django.http import HttpResponse
from .models import *

from datetime import datetime, timedelta
import pytz
import json
import pandas as pd
import numpy as np
import os
import re
import copy
import pickle
import random
from dateparser.search import search_dates
from collections import OrderedDict, Counter

from django.conf import settings
from django.views.decorators.csrf import csrf_exempt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


# Create your views here.
def event_sec_ago(d):
    '''
    event triggered how many seconds ago
    d: event dict, {'id': 'C179002853-ORNL_DAAC', 'time': '2020-10-01 20:33:45'}
    '''
    et = datetime.strptime(d['time'], "%Y-%m-%d %H:%M:%S")
    nt = datetime.now()
    return (nt-et).total_seconds() + 2 # automatically add 2 sceonds for 1/ln x weight

def rx(rcvj, key='view'):
    '''
    receiver decoder
    rcvj: received json
    key: onee of 'view', 'select', 'blog'

    rcvj[key]:
        [{'id': 'C179002853-ORNL_DAAC', 'time': '2020-10-01 20:33:45'}, 
        {'id': 'C179003246-ORNL_DAAC', 'time': '2020-09-30 05:44:45'}, 
        {'id': 'C1251051178-GES_DISC', 'time': '2020-09-30 04:38:45'}]
    '''
    ids = []
    times = []
    for e in rcvj[key]:
        ids.append(e['id'])
        times.append( 
            event_sec_ago(e))

    return ids, times
    
def lambda1(x):
    return datetime.strptime(x['time'], "%Y-%m-%d %H:%M:%S")

def time_decay_weight(l):
    '''
    calculate  weight of l
        l: a list of (id, time), ie. [('C179002853-ORNL_DAAC', 22785.289336), ('C179003246-ORNL_DAAC', 162525.289443), ('C1251051178-GES_DISC', 1.1047979187766437)]
    '''
    def reciprocal(x):
        return 1/x
    
    def ln_reciprocal(x):
        return 1/np.log10(x)
    
    def exp_ln_reciprocal(x):
        return np.exp(5/np.log(x))
    
    def xshift(x, f, shift=0):
        return f(x-shift)
    
    m = min([ time for id, time in l])
    # force first event happend at 8.77 second, which get a weight of 10.001
    l = [(id, xshift(time, exp_ln_reciprocal, m-8.77) )for id, time in l]
    l = sorted(l, key=lambda x: x[1], reverse=True)
    
    return l

def weight_normalize(l):
    '''
    normalize weight
        l: a list of (id, weight), ie. weighted [('C179002853-ORNL_DAAC', 22785.289336), ('C179003246-ORNL_DAAC', 162525.289443), ('C1251051178-GES_DISC', 166485.289502)]
                                       become   [('C1251051178-GES_DISC', 1.086733332073592), ('C179003246-ORNL_DAAC', 1.0869147017612368), ('C179002853-ORNL_DAAC', 4.3887965838556805e-05)]
    '''
    m = max([ weight for _, weight in l])
    l = [ (id, weight/m ) for id, weight in l]
    return l

def remove_duplicate(l, n=-1):
    picked = []
    picked_id = {}
    for id, score in l:                
        if id not in picked_id:
            picked.append((id, score))

        if len(picked) == n:
            break
    return picked

# for recommend_engine()
df_id = pd.read_csv(f'{settings.BASE_DIR}/rcmd/top_ids_100.csv', index_col=0)
df_score = pd.read_csv(f'{settings.BASE_DIR}/rcmd/top_scores_100.csv', index_col=0)

def recommend_engine(vlist, slist, blist, mode = 'B', N=50):
    '''
    vlist: list of tuple, dict會照發生順序排，最靠近現在的排在 0 位
           dict元素為瀏覽資料集的id:動作時間
    slist: list of tuple, dict會照發生順序排，最靠近現在的排在 0 位
           勾選資料集的id:動作時間
    blist: list of tuple, dict會照發生順序排，最靠近現在的排在 0 位
           瀏覽部落格的id:動作時間
    global variables:
        df_id:
        df_score:
    '''
    if not vlist and not slist and not blist:
        pass#return []


    # recommend routine
    def routine(xlist, aw, n=-1):
        # - reweight - #
        xlist = copy.deepcopy(xlist)
        xlist = time_decay_weight(xlist)
        xlist = weight_normalize(xlist)

        ids = [ id for id, _ in xlist]
        weights = [ weight for _, weight in xlist ]
        action_weights = np.array(weights) * np.array(aw)  

        # - weight * score - #
        score = df_score.loc[ids,:].copy()
        score = score.mul(action_weights, axis=0)
        score = score.to_numpy().flatten().tolist()

        indices = df_id.loc[ids, :].to_numpy().flatten().tolist()
        rcmd = list(zip(indices, score))
        rcmd = sorted(rcmd, key=lambda x: x[1], reverse=True)

        return [] if not rcmd else rcmd[:n]
        
    if mode == 'A':
        top3_id = []
        if vlist:
            top3_id.append(vlist[0][0])
        if slist:
            top3_id.append(slist[0][0])
        if blist:
            top3_id.append(blist[0][0])
            
        top3 = df_id.loc[top3_id, 'recommend_1'].tolist()
        
        # - remove duplicate id - #
        slist = [e for e in slist]
        vlist = [e for e in vlist if not e[0] in slist]
        blist = [e for e in blist if not e[0] in vlist]
        blist = [e for e in blist if not e[0] in slist]
        
        # - sort by time - #
        _all = slist + vlist + blist
        _all.sort(key=lambda x: x[1])
        wt = [1./i for i in range(1,len(_all)+1)]

        _all_id = [e[0] for e in _all]

        score = df_score.loc[_all_id].copy()
        score = score.mul(wt, axis=0)
        score = score.to_numpy().flatten().tolist()

        indices = df_id.loc[_all_id].to_numpy().flatten().tolist()
        rcmd = list(zip(indices, score))
        rcmd.sort(key=lambda x: x[1], reverse=True)
        rcmd = top3 + [e[0] for e in rcmd]
        rcmd = list(OrderedDict.fromkeys(rcmd)) # remove duplicate
        rcmd = rcmd[:N]
        
    elif mode == 'B':
        # action's weight #
        vw, sw, bw = 1.0, 2.0, 1.0 
        top3_bonud = 1.2
        
        # top 3 recommend   
        top3 = []
        if vlist:
            top3 = top3 + routine([vlist.pop(0)], [vw*top3_bonud], n=-1)
        if slist:
            top3 = top3 + routine([slist.pop(0)], [vw*top3_bonud], n=-1)
        if blist:
            top3 = top3 + routine([blist.pop(0)], [vw*top3_bonud], n=-1)
        
        ################################################################
            
        # user recommend
        xlist = vlist + slist + blist
        aw = [vw]*len(vlist) + [sw]*len(slist) + [bw]*len(blist)
              
        if xlist:
            rcmd = routine(xlist, aw, -1)
            # take top N - 3 in rcmd, conbine with top3, then sort again
            rcmd = top3 + remove_duplicate(rcmd, N-3)
        else:
            rcmd = top3

        rcmd = sorted(rcmd, key=lambda x: x[1], reverse=True)
        rcmd = rcmd[:N]
        rcmd = [id for id, _ in rcmd]
       
    elif mode == 'C':
        # --> same as mode 'B'
        # action's weight #
        vw, sw, bw = 1.0, 2.0, 1.0 
        top3_bonud = 1.2
        
        # top 3 recommend   
        top3 = []
        if vlist:
            top3 = top3 + routine([vlist.pop(0)], [vw*top3_bonud], n=-1)
        if slist:
            top3 = top3 + routine([slist.pop(0)], [vw*top3_bonud], n=-1)
        if blist:
            top3 = top3 + routine([blist.pop(0)], [vw*top3_bonud], n=-1)
        
        ################################################################
            
        # user recommend
        xlist = vlist + slist + blist
        aw = [vw]*len(vlist) + [sw]*len(slist) + [bw]*len(blist)

        if xlist:
            rcmd = routine(xlist, aw, -1)
            # <-- same as mode 'B'

            # sum all score
            sumscore = {}
            for id, scr in rcmd:
                s = sumscore.get(id, 0) + scr
                sumscore[id] = s

            rcmd = sorted(sumscore.items(), key=lambda x: x[1], reverse=True)
        
            # take top N - 3 in rcmd, conbine with top3, then sort again
            rcmd = top3 + remove_duplicate(rcmd, N-3)
        else:
            rcmd = top3
            
        rcmd = sorted(rcmd, key=lambda x: x[1], reverse=True)
        rcmd = rcmd[:N] 
        rcmd = [id for id, _ in rcmd]        

    return rcmd

index_to_id = pickle.load(open(f'{settings.BASE_DIR}/rcmd/index_to_id.pkl', 'rb'))
dataset_wv_norm = pickle.load(open(f'{settings.BASE_DIR}/rcmd/dataset_wv_norm.pkl', 'rb'))
wv = pickle.load(open(f'{settings.BASE_DIR}/rcmd/wordvector.pkl', 'rb'))

def search_engine(keyword, N=50):
    def row_major_vector_div_by_norm(rowvector):
        rowvector_norm = np.sum(np.abs(rowvector)**2,axis=-1)**(1./2)
        rowvector_norm = rowvector_norm.reshape((-1,1))
        rowvector_norm = np.concatenate( [rowvector_norm]*rowvector.shape[1], axis=1)
        rowvector = rowvector/rowvector_norm
        return rowvector

    punc_re = '[¿\<#í¯!°‐*�@;\\~\t•ﬂ\]½\}%¹$\)“""\(’≥μ\{\[″\xa0\u202f‘ˉ=ﬁ+±º"ó”?&—µ\xad\>,.²_\r^\n–ﬀ\|:/]'
    url = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"

    # keyword word vector
    _ = re.sub(url, ' ', keyword)
    _ = re.sub(punc_re, ' ', _)
    _ = re.sub('  ', ' ', _)
    _ = re.sub('  ', ' ', _)
    _ = re.sub('  ', ' ', _)
    keyword = re.sub('  ', ' ', _)
    keywords = keyword.split(' ')

    keyword_wv = [wv[kw].reshape((-1,1)) for kw in keywords  if kw in wv]
    if not keyword_wv:
        return []
    keyword_wv = np.concatenate(keyword_wv, axis=1)
    keyword_wv_norm = row_major_vector_div_by_norm(keyword_wv)

    # cosine similarity
    cos_similarity = dataset_wv_norm.dot(keyword_wv_norm).sum(axis=1)
    cos_similarity = list(cos_similarity)
    cos_similarity = [(index_to_id[i], _) for i, _ in enumerate(cos_similarity) if i in index_to_id]
    cos_similarity = sorted(cos_similarity, key=lambda x: x[1], reverse=True)
    rcmd = [ id for id, cos in cos_similarity[:N]] 

    return rcmd

context = pickle.load(open("keyword.pkl", "rb"))
word_to_keyword = context["word_to_keyword"]
keyword_to_id = context["keyword_to_id"]

def keyword_search(keywords, N=50):
    '''
    keywords: string containing keywords
    global variables:
        word_to_keyword: dictionary of set
        keyword_to_id: dictionary of list
    '''
    _all = []
    keywords = keywords.replace("/", " ")
    for word in keywords.lower().split():
        if word in word_to_keyword:
            for kw in word_to_keyword[word]:
                _all += keyword_to_id[kw]
    counter = Counter(_all)
    # print(counter.most_common(N))
    return [e[0] for e in counter.most_common(N)]

datetime_df = pd.read_pickle(f'{settings.BASE_DIR}/rcmd/datetime.pkl')
def temporal_filter(time_start, time_end, idList):
    TIME_MAX = pd.to_datetime("2262-01-01", format="%Y-%m-%d")
    TIME_MIN = pd.to_datetime("1678-01-01", format="%Y-%m-%d")

    if not idList:
        return []
    time_start = pd.to_datetime(time_start, format="%Y-%m-%d", errors="coerce")
    time_end = pd.to_datetime(time_end, format="%Y-%m-%d", errors="coerce")
    if pd.isnull(time_start) and pd.isnull(time_end):
        return idList
    if pd.isnull(time_start):
        time_start = TIME_MIN
    if pd.isnull(time_end):
        time_end = TIME_MAX
    if time_start>time_end:
        time_start = TIME_MIN
        time_end = TIME_MAX

    idList = datetime_df.index.intersection(idList)
    col_end = ["interval_1", "interval_3", "interval_5", "interval_7", "interval_9", "interval_11", "interval_13", "interval_15", "interval_17", "interval_19", "interval_21", "interval_23", "interval_25"]
    col_start = ["interval_0", "interval_2", "interval_4", "interval_6", "interval_8", "interval_10", "interval_12", "interval_14", "interval_16", "interval_18", "interval_20", "interval_22", "interval_24"]
    col_time = ["time_0", "time_1", "time_2", "time_3", "time_4", "time_5"]
    condition_start = datetime_df.loc[idList, col_start]<=time_end
    condition_end   = datetime_df.loc[idList, col_end]>=time_start
    condition_time1 = datetime_df.loc[idList, col_time]<=time_end 
    condition_time2 = datetime_df.loc[idList, col_time]>=time_start
    condition_time = condition_time1 & condition_time2
    condition_time["0001"] = condition_start["interval_0"] & condition_end["interval_1"]
    condition_time["0203"] = condition_start["interval_2"] & condition_end["interval_3"]
    condition_time["0405"] = condition_start["interval_4"] & condition_end["interval_5"]
    condition_time["0607"] = condition_start["interval_6"] & condition_end["interval_7"]
    condition_time["0809"] = condition_start["interval_8"] & condition_end["interval_9"]
    condition_time["1011"] = condition_start["interval_10"] & condition_end["interval_11"]
    condition_time["1213"] = condition_start["interval_12"] & condition_end["interval_13"]
    condition_time["1415"] = condition_start["interval_14"] & condition_end["interval_15"]
    condition_time["1617"] = condition_start["interval_16"] & condition_end["interval_17"]
    condition_time["1819"] = condition_start["interval_18"] & condition_end["interval_19"]
    condition_time["2021"] = condition_start["interval_20"] & condition_end["interval_21"]
    condition_time["2223"] = condition_start["interval_22"] & condition_end["interval_23"]
    condition_time["2425"] = condition_start["interval_24"] & condition_end["interval_25"]
    condition_time = condition_time.any(axis="columns")
    ret = condition_time.index[condition_time[:]]
    return list(ret)

def extract_year(text):
    dates = search_dates(text)
    if not dates:
        return []
    filtered_dates = set()
    for e in dates:
        kwords = e[0].split(",")
        date = e[1]
        for kw in kwords:
            kw = kw.rstrip().lstrip()
            if kw[:4].isdigit() and len(kw)>=4:
                filtered_dates.add(date)
    return list(filtered_dates)

def auto_time_detection(text, idList, N=3):
    if not text or not idList:
        return []
    ret = set()
    filtered_dates = extract_year(text)
    for date in filtered_dates[:N]:
        TEN_YEARS = timedelta(days=3650)
        time_start = date - TEN_YEARS
        time_end = date + TEN_YEARS
        time_start = time_start.strftime("%Y-%m-%d")
        time_end = time_end.strftime("%Y-%m-%d")
        ret = ret.union( set(temporal_filter(time_start, time_end, idList)) )
    return list(ret)

def default_recommend(N=50):
    '''
    global variables:
        keyword_to_id
    '''
    keywords = ["spectral engineering","atmospheric chemistry","oceans",
"land surface","biosphere","atmosphere"]
    _all = []
    for kw in keywords:
        _all += keyword_to_id[kw]
    return random.sample(_all, N)

@csrf_exempt
def recommend(request):
    rcv = json.loads(request.body)

    mode = rcv['mode']

    v = sorted(rcv['view'], key=lambda1, reverse=True)
    s = sorted(rcv['select'], key=lambda1, reverse=True)
    b = sorted(rcv['blog'], key=lambda1, reverse=True)



    vids, vtimes = rx(rcv, 'view')
    sids, stimes = rx(rcv, 'select')
    bids, btimes = rx(rcv, 'blog')

    # call recommantor
    vlist = [ (id, time) for id, time in zip(vids, vtimes) ]
    slist = [ (id, time) for id, time in zip(sids, stimes) ]
    blist = [ (id, time) for id, time in zip(bids, btimes) ]
    
    if not v and not s and not b:
        rr =  default_recommend(N=20)
    else:
        rr = recommend_engine(vlist=vlist, slist=slist, blist=blist,  mode = mode, N=20) 
       
    
    res = {"data":[]}
    for r in rr:
        with open(f'{settings.BASE_DIR}/rcmd/json/{r}.json', 'r') as f:
            res["data"].append( json.load(f) )
    res = json.dumps(res)
    
    return HttpResponse(res, content_type="application/json")

@csrf_exempt
def search(request):
    rcv = json.loads(request.body)
    keyword = rcv['keyword']
    start_time = rcv['start_time']
    end_time = rcv['end_time']

    # - search datasets in any time interval - #
    if not start_time and not end_time: 
        if not keyword:
            rr =  default_recommend(N=20)
        else:
            rr = keyword_search(keywords=keyword, N=20)
            rr = rr + search_engine(keyword=keyword, N=20)
    # - search datasets in specify time interval - #
    else:                               
        if not keyword:
            rr =  default_recommend(N=100)
        else:
            rr = keyword_search(keywords=keyword, N=50)
            rr = rr + search_engine(keyword=keyword, N=50)
        rr = temporal_filter(start_time, end_time, rr)
    rr = rr[:20]
    
    smart_id = auto_time_detection(keyword, rr)

    res = {'data':[]}
    for r in rr:
        with open(f'{settings.BASE_DIR}/rcmd/json/{r}.json', 'r') as f:
            js = json.load(f)
        js["smart"] = 1 if r in smart_id else 0             
        res['data'].append( js )
    res = json.dumps(res)
    
    return HttpResponse(res, content_type="application/json")
