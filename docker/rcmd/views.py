from django.shortcuts import render
from django.template.loader import get_template
from django.http import HttpResponse
from .models import *

import datetime
import pytz
import json
import pandas as pd
import numpy as np
import os
from django.conf import settings

from django.views.decorators.csrf import csrf_exempt


# Create your views here.

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
        times.append(e['time'])

    return ids, times
    


@csrf_exempt
def recommend(request):
    print(type(request.body))
    rcv = json.loads(request.body)
    print(f'get json from view, \njson: {rcv}')
    v = rcv['view']
    s = rcv['select']
    b = rcv['blog']

    vids, vtimes = rx(rcv, 'view')
    sids, stimes = rx(rcv, 'select')
    bids, btimes = rx(rcv, 'blog')

    


    # fake recommend data
    rr = vids+sids+bids
        
    res = '['
    for r in rr:
        with open(f'{settings.BASE_DIR}/rcmd/json/{r}.json', 'r') as f:
            res = res + f.read() +', '
    else:
        res = res + ']'
    return HttpResponse(res, content_type="application/json")

@csrf_exempt
def search(request):
    print(type(request.body))
    rcv = json.loads(request.body)
    print(f'get json from view, \njson: {rcv}')
    v = rcv['view']
    s = rcv['select']
    b = rcv['blog']

    vids, vtimes = rx(rcv, 'view')
    sids, stimes = rx(rcv, 'select')
    bids, btimes = rx(rcv, 'blog')

    


    # fake recommend data
    rr = vids+sids+bids
        
    res = '['
    for r in rr:
        with open(f'{settings.BASE_DIR}/rcmd/json/{r}.json', 'r') as f:
            res = res + f.read() +', '
    else:
        res = res + ']'
    return HttpResponse(res, content_type="application/json")
