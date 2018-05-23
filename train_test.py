# -*- coding: utf-8 -*-
__author__ = 'kangxun'


import numpy
#train 0.7 test 0.3
def train_produce(R,per,N,M):
    for i in xrange(N):
        for j in xrange(M):
            if R[i][j] > 0:
                n = numpy.random.random()#0-1 random
                if n > per:
                    R[i][j] = 0
    return R

def rmse_score(R,nR,N,M,e = 0,cnt = 0):
    for i in xrange(N):
        for j in xrange(M):
            if R[i][j] > 0:
                cnt += 1
                e += pow(R[i][j] - nR[i][j], 2)
    e = numpy.sqrt(e/cnt)
    return e

def test_rmse_score(data,R,nR,N,M,e = 0,cnt = 0):
    for i in xrange(N):
        for j in xrange(M):
            if R[i][j] == 0 and data[i][j] != 0:
                cnt += 1
                e += pow(data[i][j] - nR[i][j], 2)
    if cnt != 0:
        e = numpy.sqrt(e/cnt)
    else:
        e = 'error'
    return e


def matrix_factorization(R,P,Q,K,N,M,steps=5000,alpha=0.0002,beta=0.02):
    Q=Q.T
    for step in xrange(steps):
        for i in xrange(N):#user
            for j in xrange(M):#movie
                if R[i][j] > 0:
                    eij = R[i][j] - numpy.dot(P[i,:],Q[:,j])
                    for k in xrange(K):#update p\q
                        P[i][k] = P[i][k] + alpha*(2*eij*Q[k][j]-beta*P[i][k])
                        Q[k][j] = Q[k][j] + alpha*(2*eij*P[i][k]-beta*Q[k][j])
        eR = numpy.dot(P,Q)
        e = 0
        for i in xrange(N):
            for j in xrange(M):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - numpy.dot(P[i,:],Q[:,j]), 2)
                    for k in xrange(K):
                        e = e + (beta/2) *(pow(P[i][k],2) + pow(Q[k][j],2))
        print "step %d error %f" % (step,e)
        if e < 0.001:
            break
    return P,Q.T

def recommend(data,nR,N,M,top=1):
    for i in xrange(N):#user
        score = {}
        for j in xrange(M):#movie
            if data[i][j] == 0:
                score[str(j)] = nR[i][j]
        score = sorted(score.items(),key = lambda x:x[1],reverse = True)
        score = dict(score)
        #print score
        if len(score) < top:
            return 'error'
        cnt = 0
        for key in score.keys():
            cnt += 1
            print "user %d recommend movie %s" % (i,key)
            if cnt == top:
                break


data = [
    [5,3,0,1],
    [4,0,0,1],
    [1,1,0,5],
    [1,0,0,4],
    [0,1,5,4],
]
data = numpy.array(data)
N = data.shape[0]#user
M = data.shape[1]#movie
K = 2

#train_R =  numpy.empty_like(R)#the same shape as the R
train = data.copy()#copy R array
train = train_produce(train,0.7,N,M)
#train MF

train_P = numpy.random.rand(N,K)
train_Q = numpy.random.rand(M,K)
train_nP,train_nQ = matrix_factorization(train,train_P,train_Q,K,N,M)
train_nR = numpy.dot(train_nP,train_nQ.T)
#rmse
train_err = rmse_score(train,train_nR,N,M)
test_err = test_rmse_score(data,train,train_nR,N,M)
print train_err
print test_err
print data
print train
print train_nR

recommend(data,train_nR,N,M)
#print ' '
#print nR1