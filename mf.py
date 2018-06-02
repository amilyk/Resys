# -*- coding: utf-8 -*-
__author__ = 'kangxun'

import pandas as pd
import numpy as np
import time
import evaluation
#train_RMSE train与 P*Q得到的预测矩阵train'之间评分差值(train 非0部分)
#test_RMSE test与 train'之间差值(test 非0部分)

def loadData():
    train = pd.io.parsers.read_csv("train.csv",
    names=['user_id', 'movie_id', 'rating'],
    engine='python', delimiter=',')
    test = pd.io.parsers.read_csv("test.csv",
    names=['user_id', 'movie_id', 'rating'],
    engine='python', delimiter=',')

    #为了确保 (train->train')得到的train'与 test 维度一致的.
    #print np.max(train.user_id.values)
    #print np.max(train.movie_id.values)
    #print np.max(test.user_id.values)
    #print np.max(test.movie_id.values)
    train_m = np.max(train.user_id.values)
    train_n = np.max(train.movie_id.values)
    test_m = np.max(test.user_id.values)
    test_n = np.max(test.movie_id.values)


    n = max(np.max(train.user_id.values),np.max(test.user_id.values))
    m = max(np.max(train.movie_id.values),np.max(test.movie_id.values))
    train_ratings = np.ndarray(shape=(n,m),dtype=np.uint8)
    test_ratings = np.ndarray(shape=(n,m),dtype=np.uint8)
    train_ratings[train.user_id.values-1,train.movie_id.values-1] = train.rating.values
    test_ratings[test.user_id.values-1,test.movie_id.values-1] = test.rating.values
    #评分为空已经默认填0
    #print np.max(train.movie_id.values),np.max(test.movie_id.values)
    #print test_ratings[942][1681]
    #return ratings_mat
    return train_ratings,test_ratings


def matrix_factorization(R,P,Q,K,N,M,steps=5000,alpha=0.0002,beta=0.02):#改成随机梯度下降
    Q=Q.T
    for step in xrange(steps):
        start_time = time.time()
        for i in xrange(N):#user
            for j in xrange(M):#movie
                if R[i][j] > 0:
                    eij = R[i][j] - np.dot(P[i,:],Q[:,j])
                    for k in xrange(K):#update p\q
                        P[i][k] = P[i][k] + alpha*(2*eij*Q[k][j]-beta*P[i][k])
                        Q[k][j] = Q[k][j] + alpha*(2*eij*P[i][k]-beta*Q[k][j])
        eR = np.dot(P,Q)
        e = 0
        for i in xrange(N):
            for j in xrange(M):
                if R[i][j] > 0:
                    e = e + (R[i][j] - np.dot(P[i,:],Q[:,j])) ** 2
                    for k in xrange(K):
                        e = e + (beta/2) *( P[i][k]**2 + Q[k][j]**2)
        print "step %d, error %f, cost time %f" % (step,e,time.time()-start_time)
        if e < 0.001:
            break
    return P,Q.T


##test
def testData():
    data = pd.io.parsers.read_csv("ml-100k/u.data",
    names=['user_id', 'movie_id', 'rating', 'time'],
    engine='python', delimiter='\t')
    # movie_data = pd.io.parsers.read_csv("movies.dat",
    # names=['movie_id', 'title', 'genre'],
    # engine='python', delimiter='::')
    ratings_mat = np.ndarray(shape=(np.max(data.user_id.values), np.max(data.movie_id.values)),dtype=np.uint8)#user * movie
    ratings_mat[data.user_id.values-1,data.movie_id.values-1] = data.rating.values
    return ratings_mat
print "loading "
R = testData()
N = R.shape[0]
M = R.shape[1]
K = 2
print "Start training MF."
P = np.random.rand(N,K)
Q = np.random.rand(M,K)
P,Q = matrix_factorization(R,P,Q,K,N,M,steps=10)
nR = np.dot(P,Q.T)
print "Start Evaluation."
train_err = evaluation.rmse_Score(R,nR,N,M)
print train_err


#print "loading train and test set."
#traindata,testdata = loadData()
#N = traindata.shape[0]#user
#M = traindata.shape[1]#movie
#K = 2

#mf
#print "Start training MF."
#train_P = np.random.rand(N,K)
#train_Q = np.random.rand(M,K)
#train_nP,train_nQ = matrix_factorization(traindata,train_P,train_Q,K,N,M,steps=10)
#train_nR = np.dot(train_nP,train_nQ.T)

#rmse
#print "Start Evaluation."
#train_err = evaluation.rmse_Score(traindata,train_nR,N,M)
#test_err = evaluation.rmse_Score(testdata,train_nR,N,M)
#print train_err,test_err



