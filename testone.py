# -*- coding: utf-8 -*-
__author__ = 'kangxun'

import pandas as pd
import numpy as np
import time
import evaluation
#train_RMSE train与 P*Q得到的预测矩阵train'之间评分差值(train 非0部分)
#test_RMSE test与 train'之间差值(test 非0部分)

def loadData():
    train = pd.io.parsers.read_csv("data/train.csv",
    names=['user_id', 'movie_id', 'rating'],
    engine='python', delimiter=',')
    test = pd.io.parsers.read_csv("data/test.csv",
    names=['user_id', 'movie_id', 'rating'],
    engine='python', delimiter=',')

    #80%train 20test
    print test.shape
    print train.shape


    #为了确保 (train->train')得到的train'与 test 维度一致的.
    print np.max(train.user_id.values)
    print np.max(train.movie_id.values)
    print np.max(test.user_id.values)
    print np.max(test.movie_id.values)

    n = max(np.max(train.user_id.values),np.max(test.user_id.values))
    m = max(np.max(train.movie_id.values),np.max(test.movie_id.values))
    train_ratings = np.ndarray(shape=(n,m),dtype=np.float)
    test_ratings = np.ndarray(shape=(n,m),dtype=np.float)
    train_ratings[train.user_id.values-1,train.movie_id.values-1] = train.rating.values
    test_ratings[test.user_id.values-1,test.movie_id.values-1] = test.rating.values
    #print train_ratings.shape,test_ratings.shape
    #print np.isnan(test_ratings).any()
    #print np.isnan(train_ratings).any()
    #print len(train_ratings[np.nonzero(train_ratings)])
    #print len(test_ratings[np.nonzero(test_ratings)])

    print len(set(train.user_id)),len(train.user_id)
    print len(set(test.user_id)),len(test.user_id)
    print len(set(train.movie_id)),len(train.movie_id)
    print len(set(test.movie_id)),len(test.movie_id)
    print max(test.movie_id),max(train.movie_id)
    print max(train.user_id),max(test.user_id)




loadData()