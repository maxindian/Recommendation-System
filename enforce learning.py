import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helper
from six.moves import urllib
from tensorflow.contrib import learn
from numpy.random import RandomState
from scipy import sparse
from scipy.sparse import lil_matrix
import autoencoder
from PMF import PMF
from evaluation import RMSE
from PMF_12 import PMF
from BPMF import BPMF 
import pandas as pd 
rating_data_file='ml-1m/ml-1m/ratings.dat'
ratings = data_helper.load_movielens_ratings(rating_data_file)

n_user = max(ratings[:, 0])
n_item = max(ratings[:, 1])

# shift user_id & movie_id by 1. let user_id & movie_id start from 0
ratings[:, (0, 1)] -= 1
train_preds = pd.read_table('11.txt')
train_preds = np.array(train_preds)
print(train_preds.shape)
# print(ratings[:, 2].shape)

# train_rmse = RMSE(train_preds, ratings[:, 2])
# print(train_rmse)