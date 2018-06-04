# -*- coding: utf-8 -*-
__author__ = 'kangxun'

#不足:随机梯度下降+评分没有归一化到1-5
import numpy
import pandas as pd
import time
#train 0.7 test 0.3
def train_produce(R,per,N,M):
    # for i in xrange(N):
    #     for j in xrange(M):
    #         if R[i][j] > 0:
    #             n = numpy.random.random()#0-1 random
    #             print n
    #             if n > per:
    #                 R[i][j] = 0
    selected = numpy.random.rand(N,M)
    for i in xrange(N):
        for j in xrange(M):
            if R[i][j] > 0:
                n = selected[i][j]
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


def matrix_factorization(R,P,Q,K,N,M,steps=5000,alpha=0.0002,beta=0.02):#改成随机梯度下降
    Q=Q.T
    for step in xrange(steps):
        start_time = time.time()
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
                    e = e + (R[i][j] - numpy.dot(P[i,:],Q[:,j])) ** 2
                    for k in xrange(K):
                        e = e + (beta/2) *( P[i][k]**2 + Q[k][j]**2)
        print "step %d, error %f, cost time %f" % (step,e,time.time()-start_time)
        if e < 0.001:
            break
    return P,Q.T

def recommend(R,nR,N,M,top=1):#根据预测后的评分,对每一个 user 推荐没评分电影中预测评分最高的 top 电影
    for i in xrange(N):#user
        score = {}
        for j in xrange(M):#movie
            if R[i][j] == 0:
                score[str(j)] = nR[i][j]
        score = sorted(score.items(),key = lambda x:x[1],reverse = True)
        score = dict(score)
        #print score
        if len(score) < top:
            return 'error'
        cnt = 0
        for key in score.keys():
            cnt += 1
            # print "user %d recommend movie %s" % (i,key)
            if cnt == top:
                break

def loadData():
    data = pd.io.parsers.read_csv("ratings.dat",
    names=['user_id', 'movie_id', 'rating', 'time'],
    engine='python', delimiter='::')
    # movie_data = pd.io.parsers.read_csv("movies.dat",
    # names=['movie_id', 'title', 'genre'],
    # engine='python', delimiter='::')
    ratings_mat = numpy.ndarray(shape=(numpy.max(data.user_id.values), numpy.max(data.movie_id.values)),dtype=numpy.uint8)#user * movie
    ratings_mat[data.user_id.values-1,data.movie_id.values-1] = data.rating.values
    return ratings_mat

mydata = loadData()
print mydata
N = mydata.shape[0]#user
M = mydata.shape[1]#movie
K = 2

#train_R =  numpy.empty_like(R)#the same shape as the R
train = mydata.copy()#copy R array
print "Generating training set."
train = train_produce(train,0.7,N,M)
#train MF
print "Start training MF."
train_P = numpy.random.rand(N,K)
train_Q = numpy.random.rand(M,K)
train_nP,train_nQ = matrix_factorization(train,train_P,train_Q,K,N,M,steps=10)
train_nR = numpy.dot(train_nP,train_nQ.T)
#rmse
print "Start Evaluation."
train_err = rmse_score(train,train_nR,N,M)
test_err = test_rmse_score(mydata,train,train_nR,N,M)
print train_err
print test_err

recommend(mydata,train_nR,N,M)
#print ' '
#print nR1