# -*- coding: utf-8 -*-
__author__ = 'kangxun'
def loadExData():
    return [[4,4,0,2,2],
            [4,0,0,3,3],
            [4,0,0,1,1],
            [1,1,1,2,0],
            [2,2,2,0,0],
            [1,1,1,0,0],
            [5,5,5,0,0]]
data = loadExData()
from numpy import *
from numpy import linalg as la
def eulidSim(inA,inB):
    return 1.0/(1.0+la.norm(inA-inB))
def pearsSim(inA,inB):
    if len(inA) < 3:#??
        return 0
    return 0.5-0.5*corrcoef(inA,inB,rowvar = 0)[0][1]
def cosSim(inA,inB):
    num = float(inA.T*inB)#AB内积
    denom = la.norm(inA)*la.norm(inB)#A与B范数(模)乘积
    return 0.5+0.5*(num/denom)#本来-1~1 归一化0~1

###?
def recommend(dataMat,user,N = 3,simMeas=cosSim,estMethod=standEst):
    return



