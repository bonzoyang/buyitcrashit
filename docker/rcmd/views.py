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
@csrf_exempt
def recommend(request):
    print(type(request.body))
    rcv = json.loads(request.body)
    print(f'get json from view, \njson: {rcv}')
    v = rcv['view']
    s = rcv['select']
    b = rcv['blog']


    # fake recommend data
    rr = ['C179002853-ORNL_DAAC', 'C179003246-ORNL_DAAC', 'C1251051178-GES_DISC',
    'C1577578302-SEDAC', 'C1000001080-LARC_ASDC', 'C1674784946-PODAAC', 'C1237113469-GES_DISC',
    'C1235316217-GES_DISC', 'C1276812927-GES_DISC', 'C1274767826-GES_DISC', 'C1274764823-GES_DISC', 'C191032907-LARC',
    'C179124085-ORNL_DAAC', 'C1906827740-ORNL_DAAC', 'C1214474243-ASF', 'C7611230-LARC_ASDC',
    'C1684267683-PODAAC', 'C1597990346-NOAA_NCEI', 'C1625128618-GHRC_CLOUD', 'C1898287521-GHRC_CLOUD']
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


    # fake recommend data
    rr = ['C179002853-ORNL_DAAC', 'C179003246-ORNL_DAAC', 'C1251051178-GES_DISC',
    'C1577578302-SEDAC', 'C1000001080-LARC_ASDC', 'C1674784946-PODAAC', 'C1237113469-GES_DISC',
    'C1235316217-GES_DISC', 'C1276812927-GES_DISC', 'C1274767826-GES_DISC', 'C1274764823-GES_DISC', 'C191032907-LARC',
    'C179124085-ORNL_DAAC', 'C1906827740-ORNL_DAAC', 'C1214474243-ASF', 'C7611230-LARC_ASDC',
    'C1684267683-PODAAC', 'C1597990346-NOAA_NCEI', 'C1625128618-GHRC_CLOUD', 'C1898287521-GHRC_CLOUD']
    res = '['
    for r in rr:
        with open(f'{settings.BASE_DIR}/rcmd/json/{r}.json', 'r') as f:
            res = res + f.read() +', '
    else:
        res = res + ']'
    return HttpResponse(res, content_type="application/json")
