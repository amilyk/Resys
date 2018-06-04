__author__ = 'fanshen.fs'

import numpy as np

def get_AP(k,ideal,test):#top k ideal实际数据集 test:predict
    """
        compute AP
    """
    ideal=set(ideal)
    accumulation=0.0
    count=0
    for i in range(len(test)):
        if i>=k:
            break
        if test[i] in ideal:
            count+=1
            accumulation+=count/(i+1.0)
    m=len(ideal)
    n=k
    x=0
    if m>n:
        x=n
    else:
        x=m
    if x==0:
        return 0
    return accumulation/x


def get_MAP(k,ideal_map,test_map):
    """
        compute MAP
    """
    accumulation=0.0
    for key in ideal_map.keys():
        accumulation+=get_AP(k, ideal_map[key], test_map[key])
    if len(ideal_map)==0:
        return 0
    return accumulation/len(ideal_map)

def rmse_Score(R,nR,N,M,e = 0,cnt = 0):
    for i in xrange(N):
        for j in xrange(M):
            if R[i][j] > 0:
                cnt += 1
                e += pow(R[i][j] - nR[i][j], 2)
    e = np.sqrt(e/cnt)
    return e