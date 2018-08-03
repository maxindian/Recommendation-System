"""
Reference paper:
    "Probabilistic Matrix Factorization"
    R. Salakhutdinov and A.Mnih.
    Neural Information Processing Systems 21 (NIPS 2008). Jan. 2008.

Reference Matlab code: http://www.cs.toronto.edu/~rsalakhu/BPMF.html
"""
# coding = utf-8
import logging
from six.moves import xrange

import numpy as np
from numpy.random import RandomState
from base import ModelBase
from exceptions import NotFittedError
from validation import check_ratings
from evaluation import RMSE


logger = logging.getLogger(__name__)


class PMF(ModelBase):
    """Probabilistic Matrix Factorization
    """

    def __init__(self, n_user, n_item, n_feature,U,V,batch_size=1e5, epsilon=50.0,
                 momentum=0.8, seed=None, reg=1e-2, converge=1e-5,
                 max_rating=None, min_rating=None):

        super(PMF, self).__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.n_feature = n_feature
        self.U = U
        self.V = V
        # print(self.U)
        # print(self.V)

        self.random_state = RandomState(seed)

        # batch size
        self.batch_size = batch_size

        # learning rate
        self.epsilon = float(epsilon)
        self.momentum = float(momentum)
        # regularization parameter
        self.reg = reg
        self.converge = converge
        self.max_rating = float(max_rating) \
            if max_rating is not None else max_rating
        self.min_rating = float(min_rating) \
            if min_rating is not None else min_rating

        # data state
        self.mean_rating_ = None
        # user/item features
        # self.user_features_ = 0.1 * self.random_state.rand(n_user, n_feature)
        # self.item_features_ = 0.1 * self.random_state.rand(n_item, n_feature)
        self.user_features_ = U
        self.item_features_ = V

    def fit(self, ratings, n_iters=50):
        #print(1)
        check_ratings(ratings, self.n_user, self.n_item,
                      self.max_rating, self.min_rating)
        #print(2)
        self.mean_rating_ = np.mean(ratings[:, 2])
        #print(self.mean_rating_)
        last_rmse = None
        #print(3)
        batch_num = int(np.ceil(float(ratings.shape[0]/ self.batch_size)))
        #print(batch_num)
        logger.debug("batch count = %d", batch_num + 1)
        #print(4)

        # momentum
        u_feature_mom = np.zeros((self.n_user, self.n_feature))
        #print(u_feature_mom)
        i_feature_mom = np.zeros((self.n_item, self.n_feature))
        #print(i_feature_mom)
        # gradient
        u_feature_grads = np.zeros((self.n_user, self.n_feature))
        #print(u_feature_grads)
        i_feature_grads = np.zeros((self.n_item, self.n_feature))
        #print(i_feature_grads)
        k = 0
        m = 0
        for iteration in xrange(n_iters):
            logger.debug("iteration %d...", iteration)
            #print(5)
            self.random_state.shuffle(ratings)
            #print(6)
            for batch in xrange(batch_num):
                start_idx = int(batch * self.batch_size)
                #print(start_idx)
                end_idx = int((batch + 1) * self.batch_size)
                #print(end_idx)
                data = ratings[start_idx:end_idx]
                #print(data)
                #print(data.take(0, axis=1))
                # compute gradient
                u_features = self.user_features_.take(
                    data.take(0, axis=1), axis=0)  #data.take(0, axis=1)表示取用户值，user_features_.take表示取这些用户对应的特征值
                #print(k)
                #print(u_features)
                i_features = self.item_features_.take(
                    data.take(1, axis=1), axis=0)
                #print(data.take(1, axis=1))
                #print(m)
                #print(i_features)
                k = k+1
                m = m+1
                preds = np.sum(u_features * i_features, 1)    #即可得到用户i对item i 的ratings
                #print(preds)
                errs = preds - (data.take(2, axis=1) - self.mean_rating_)   #为什么要减去平均值呢
                #print(errs)
                # print(errs)
                # print("eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
                err_mat = np.tile(2 * errs, (self.n_feature, 1)).T
                #print(err_mat)
                u_grads = i_features * err_mat + self.reg * u_features
                i_grads = u_features * err_mat + self.reg * i_features
                #print(u_grads)
                #print(i_grads)
                u_feature_grads.fill(0.0)
                #print(u_feature_grads)
                i_feature_grads.fill(0.0)
                #print(i_feature_grads)
                for i in xrange(data.shape[0]):
                    row = data.take(i, axis=0)
                    #print(row)
                    u_feature_grads[row[0], :] += u_grads.take(i, axis=0)
                    i_feature_grads[row[1], :] += i_grads.take(i, axis=0)
                    #print(u_feature_grads)
                    #print(i_feature_grads)
                # update momentum
                u_feature_mom = (self.momentum * u_feature_mom) + \
                    ((self.epsilon / data.shape[0]) * u_feature_grads)
                #print(u_feature_mom)
                i_feature_mom = (self.momentum * i_feature_mom) + \
                    ((self.epsilon / data.shape[0]) * i_feature_grads)
                #print(i_feature_mom)
                # update latent variables
                self.user_features_ -= u_feature_mom
                #print(self.user_features_)
                self.item_features_ -= i_feature_mom
                #print(self.item_features_)
           
            # compute RMSE

            train_preds = self.predict(ratings[:, :2])
            #print(train_preds)
            train_rmse = RMSE(train_preds, ratings[:, 2])
            #print(train_rmse)
            logger.info("iter: %d, train RMSE: %.6f", iteration, train_rmse)

            # stop when converge
            if last_rmse and abs(train_rmse - last_rmse) < self.converge:
                logger.info('converges at iteration %d. stop.', iteration)
                break
            else:
                last_rmse = train_rmse
        return self

    def predict(self, data):

        if not self.mean_rating_:
            raise NotFittedError()

        u_features = self.user_features_.take(data.take(0, axis=1), axis=0)
        #print(u_features)
        i_features = self.item_features_.take(data.take(1, axis=1), axis=0)
        #print(i_features)
        preds = np.sum(u_features * i_features, 1) + self.mean_rating_
        #print(preds)

        if self.max_rating:
            preds[preds > self.max_rating] = self.max_rating

        if self.min_rating:
            preds[preds < self.min_rating] = self.min_rating
        return preds
